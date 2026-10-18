#!/usr/bin/env python3
"""Validate a kernel pack against docs/pack-format.md. Standard library only.

  python3 -I tools/check_pack.py <pack dir> [--ggml model.bin] [--no-files]

The reference for what a native loader must refuse: an unknown schema major,
an unknown kind or variant, a path outside designs/ (or a symlink), a file
whose size or sha256 does not match, a pack id that is not the hash of its
designs, a ladder step whose roles name a missing design, a memlock figure
that does not follow its rule, a selection table that does not follow its
rule. With --ggml it also checks the model file the way a loader would: the
encoder hparams must equal the pack's, every encoder tensor type must be in
weight_types, and a different sha256 is only reported.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

SCHEMA_MAJOR = ("xdna-whisper-pack", 1)
KINDS = {"gemm": {None}, "softmax": {"s3", "s4w"}, "gelu": {"g4r", "g4f"}}
ROLES = {"conv1": "gemm", "conv2": "gemm", "gelu": "gelu", "proj": "gemm",
         "qk": "gemm", "softmax": "softmax", "av": "gemm", "fc2": "gemm"}
MLP_ROLES = {"plain": {"fc1": "gemm"}, "fused": {"fc1_gelu": "gelu"}}
HP = ("n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer")

errors, notes = [], []


def err(msg):
    errors.append(msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_path(pack, rel):
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or not rel.startswith("designs/") or len(p.parts) != 2:
        err(f"{rel}: path must be designs/<file>")
        return None
    full = pack / p
    if full.is_symlink():
        err(f"{rel}: symlink")
        return None
    return full


def ggml_header(path):
    """hparams and encoder tensor types of a whisper.cpp ggml file."""
    names = ("n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
             "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer", "n_mels", "ftype")
    sizes = {0: 4.0, 1: 2.0, 8: 34 / 32}
    tname = {0: "f32", 1: "f16", 8: "q8_0"}
    types = set()
    with open(path, "rb") as f:
        if struct.unpack("<I", f.read(4))[0] != 0x67676D6C:
            raise ValueError("not a ggml whisper file")
        hp = dict(zip(names, struct.unpack("<11i", f.read(44))))
        n_mel, n_fft = struct.unpack("<2i", f.read(8))
        f.seek(n_mel * n_fft * 4, 1)
        for _ in range(struct.unpack("<i", f.read(4))[0]):
            f.seek(struct.unpack("<I", f.read(4))[0], 1)
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            nd, ln, tt = struct.unpack("<3i", head)
            n = math.prod(struct.unpack(f"<{nd}i", f.read(4 * nd)))
            name = f.read(ln).decode()
            if name.startswith("encoder."):
                types.add(tname.get(tt, f"type{tt}"))
            if tt not in sizes:
                raise ValueError(f"{name}: ggml type {tt}")
            f.seek(int(n * sizes[tt]), 1)
    return hp, types


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack", type=Path)
    ap.add_argument("--ggml", type=Path)
    ap.add_argument("--no-files", action="store_true",
                    help="manifest only (e.g. CI without the overlays)")
    a = ap.parse_args()
    m = json.loads((a.pack / "manifest.json").read_text())

    name, _, major = m.get("schema", "").partition("/")
    if (name, int(major or 0)) != SCHEMA_MAJOR:
        sys.exit(f"unknown schema {m.get('schema')!r}; this checker knows {SCHEMA_MAJOR}")

    designs = m["designs"]
    for dn, d in designs.items():
        kind, variant = d.get("kind"), d.get("variant")
        if kind not in KINDS or variant not in KINDS[kind]:
            err(f"{dn}: unknown kind/variant {kind}/{variant}")
        for key in ("xclbin", "insts"):
            f = d[key]
            full = check_path(a.pack, f["path"])
            if full is None or a.no_files:
                continue
            if not full.is_file():
                err(f"{f['path']}: missing")
            elif full.stat().st_size != f["bytes"]:
                err(f"{f['path']}: {full.stat().st_size} bytes, manifest says {f['bytes']}")
            elif sha256(full) != f["sha256"]:
                err(f"{f['path']}: sha256 mismatch")

    content = hashlib.sha256("".join(designs[n]["xclbin"]["sha256"] + designs[n]["insts"]["sha256"]
                                     for n in sorted(designs)).encode()).hexdigest()
    want_id = f"{m['model']['name']}-{m['device']['generation']}-{content[:12]}"
    if m["pack_id"] != want_id:
        err(f"pack_id {m['pack_id']} is not the hash of its designs ({want_id})")

    mem = m["memory"]
    floor = max(mem["heap_bytes"], mem["max_bo_bytes"])
    if mem["memlock_min_bytes"] != 1 << (floor - 1).bit_length():
        err(f"memlock_min_bytes {mem['memlock_min_bytes']} does not follow its rule")

    sel = m["selection"]
    used = set()
    prev = 0
    for step in m["ladder"]:
        ctx = step["ctx"]
        if ctx <= prev or ctx > m["model"]["n_audio_ctx"]:
            err(f"ladder ctx {ctx}: not ascending or above n_audio_ctx")
        prev = ctx
        full_ctx = ctx == m["model"]["n_audio_ctx"]
        secs = m["model"]["n_audio_ctx"] / sel["positions_per_sec"] if full_ctx else \
            (ctx - sel["margin_positions"]) / (sel["positions_per_sec"] * (1 + sel["margin_ratio"]))
        if abs(step["max_audio_secs"] - round(secs, 3)) > 1e-6:
            err(f"ctx {ctx}: max_audio_secs {step['max_audio_secs']} != rule {secs:.3f}")
        roles = step["roles"]
        path = step.get("mlp", {}).get("path")
        if path not in MLP_ROLES:
            err(f"ctx {ctx}: unknown mlp path {path!r}")
            continue
        want = dict(ROLES, **MLP_ROLES[path])
        if set(roles) != set(want):
            err(f"ctx {ctx}: roles {sorted(roles)} != {sorted(want)}")
        for role, spec in roles.items():
            dn = spec.get("design")
            if dn not in designs:
                err(f"ctx {ctx} {role}: design {dn} not in the pack")
            elif want.get(role) and designs[dn]["kind"] != want[role]:
                err(f"ctx {ctx} {role}: {dn} is a {designs[dn]['kind']}, want {want[role]}")
            used.add(dn)
    for dn in sorted(set(designs) - used):
        err(f"{dn}: in the pack but used by no ladder step")

    if a.ggml:
        hp, types = ggml_header(a.ggml)
        for k in HP:
            if hp[k] != m["model"][k]:
                err(f"{a.ggml.name}: {k}={hp[k]}, pack wants {m['model'][k]}")
        bad = sorted(types - set(m["model"]["weight_types"]) - {"f32"})
        if bad:
            err(f"{a.ggml.name}: encoder weight types {bad} not in {m['model']['weight_types']}")
        if not errors and sha256(a.ggml) != m["model"]["validated_with"]["sha256"]:
            notes.append(f"{a.ggml.name}: not the file this pack was validated with "
                         f"({m['model']['validated_with']['file']}) -- fine for the same architecture")

    for n in notes:
        print(f"note: {n}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)
    print(f"ok: {m['pack_id']} ({len(designs)} designs, ladder "
          f"{[s['ctx'] for s in m['ladder']]}, memlock >= {mem['memlock_min_bytes'] >> 20} MiB)")


if __name__ == "__main__":
    main()
