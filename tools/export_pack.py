#!/usr/bin/env python3
"""Export a kernel pack: the compiled overlays one model needs on one NPU
generation, plus the manifest a native runtime reads instead of this repo's
Python. Format and semantics: docs/pack-format.md.

Runs in the service image with stt stopped (it opens the NPU to verify):

  refenv/bin/python tools/export_pack.py --ggml ggml-large-v3-turbo.bin \\
      --name large-v3-turbo --out /probe/packs [--vcs-ref SHA]
  refenv/bin/python tools/export_pack.py --ggml ggml-tiny.bin --name tiny \\
      --ladder 512 --out /probe/packs --fixtures tests/data/jfk.wav

Every design is derived with the same functions RawRT uses to open it
(padded_shape / compile_shape / compile_rowop / chain_designs), compiled if
missing, then checked against what one real encoder pass per ladder step
actually opens on the device -- weights from the ggml file, which is the
native path. --fixtures also writes per-stage golden dumps of one pass at the
first ladder step, so the host side can be tested without an NPU.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import rawxrt
from ggml_weights import load_encoder

SCHEMA = "xdna-whisper-pack/1"
# The first mapping XRT makes on a fresh device handle, 64 MiB, before any BO
# of ours (measured 2026-10-17: the EAGAIN at memlock < 64 MiB names it).
HEAP_BYTES = 64 << 20
DEFAULT_LADDER = (512, 768, 1024, 1280, 1500)
# ctx for a clip of s seconds: ceil(s * 50 * (1 + ratio)) + positions, onto
# the ladder -- the rule serve/ uses for NPU_STT_CTX_FROM_AUDIO.
SELECTION = {"positions_per_sec": 50, "margin_ratio": 0.1, "margin_positions": 16}
WEIGHT_TYPES = ["f16", "q8_0"]
# kernel(opcode, insts, n_insts, buffers...): argument index of each operand
KERNEL_ARGS = {"gemm": {"insts": 1, "a": 3, "b": 4, "c": 5},
               "rowop": {"insts": 1, "in": 3, "out": 4}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gemm(M, K, N, dtype_out="f32", bias_folded=False):
    """A GEMM role: the design RawRT.mm_start would open, with its operands."""
    Mp, Kp, Np, TM, TK, TN = rawxrt.padded_shape(M, K, N)
    name = rawxrt.compile_shape(Mp, Kp, Np, TM, TK, TN, verbose=False,
                                dtype_out=dtype_out)
    return name, {
        "M": M, "K": K, "N": N, "Mp": Mp, "Kp": Kp, "Np": Np,
        "tiles": [TM, TK, TN], "pad": {"M": TM * 8, "K": TK, "N": TN * 4},
        "bias_folded": bias_folded,
        "a": {"layout": "row_major_bf16", "rows": Mp, "cols": Kp},
        "b": {"layout": "row_major_bf16", "rows": Kp, "cols": Np},
        "c": {"layout": f"row_major_{dtype_out}", "rows": Mp, "cols": Np},
    }


def roles_for(ctx, hp):
    n_mels, d, nh = hp["n_mels"], hp["n_audio_state"], hp["n_audio_head"]
    hd, d_ff = d // nh, 4 * d
    R, designs = {}, {}

    def add(role, name, spec, kind):
        R[role] = dict(design=name, **spec)
        designs.setdefault(name, kind)

    add("conv1", *gemm(2 * ctx, 3 * n_mels, d), "gemm")
    add("conv2", *gemm(ctx, 3 * d, d), "gemm")
    g = rawxrt.compile_rowop(rawxrt.RawRT.GELU_OP, rawxrt.RawRT.GELU_ROWS,
                             rawxrt.RawRT.GELU_COLS)
    add("gelu", g, {"rows": rawxrt.RawRT.GELU_ROWS, "cols": rawxrt.RawRT.GELU_COLS,
                    "in": "bf16", "out": "bf16",
                    "chunking": "flatten, chunks of rows*cols, zero-pad the last"},
        "gelu")
    # q, k (zero bias), v, out and every fc1 column chunk share this design
    add("proj", *gemm(ctx, d + 1, d, bias_folded=True), "gemm")
    ch = rawxrt.chain_designs(nh, ctx, hd)
    Mp, Np, aN = ch["Mp"], ch["Np"], ch["aN"]
    qk_name = ch["qk"]
    add("qk", qk_name, {
        "Mp": Mp, "Kp": hd, "Np": Np, "scale": "head_dim ** -0.25 on both q and k",
        "a": {"layout": "row_major_bf16", "rows": Mp, "cols": hd,
              "stack": "per head", "head_stride_bytes": Mp * hd * 2},
        "b": {"layout": "row_major_bf16", "rows": hd, "cols": Np,
              "stack": "per head", "head_stride_bytes": hd * Np * 2},
        "c": {"layout": "row_major_bf16", "rows": Mp, "cols": Np,
              "into": "softmax.in", "head_stride_bytes": Mp * Np * 2}}, "gemm")
    add("softmax", ch["softmax"], {
        "rows": nh * Mp, "cols": Np, "valid_cols": ch["softmax_window"],
        "in": "bf16", "out": "bf16", "head_stride_bytes": Mp * Np * 2,
        "buffer_bytes": nh * Mp * Np * 2}, "softmax")
    add("av", ch["av"], {
        "Mp": Mp, "Kp": Np, "Np": aN,
        "a": {"layout": "row_major_bf16", "rows": Mp, "cols": Np,
              "from": "softmax.out", "head_stride_bytes": Mp * Np * 2},
        "b": {"layout": "row_major_bf16", "rows": Np, "cols": aN,
              "stack": "per head", "head_stride_bytes": Mp * aN * 2},
        "c": {"layout": "row_major_f32", "rows": Mp, "cols": aN,
              "stack": "per head", "head_stride_bytes": Mp * aN * 4}}, "gemm")
    fc2_name, fc2 = gemm(ctx, d_ff, d)
    import npu_whisper_encoder as E
    if d_ff <= E.MAX_GEMM_N:
        # prepare_mlp_chunks leaves an fc1 that compiles alone: one GEMM, bias
        # added on the host, the shared GELU rowop, then fc2
        mlp = {"path": "plain"}
        add("fc1", *gemm(ctx, d, d_ff), "gemm")
    else:
        # fc1 in column chunks of width d on the proj design, each chunk's f32
        # result GELU'd on the device straight into its band of fc2's A
        fc1 = R["proj"]
        mlp = {"path": "fused", "chunks": d_ff // d}
        gf = rawxrt.compile_rowop(rawxrt.RawRT.GELU_F32_OP, fc2["Mp"], fc1["Np"],
                                  out_cols=fc2["Kp"])
        add("fc1_gelu", gf, {
            "rows": fc2["Mp"], "cols": fc1["Np"], "out_cols": fc2["Kp"],
            "in": "f32", "out": "bf16", "chunks": d_ff // d,
            "band": {"into": "fc2.a", "offset_bytes_per_chunk": fc1["Np"] * 2,
                     "row_stride_bytes": fc2["Kp"] * 2}}, "gelu")
    add("fc2", fc2_name, fc2, "gemm")
    attn = {"Mp": Mp, "Np": Np, "aN": aN, "softmax_window": ch["softmax_window"]}
    return R, designs, attn, mlp


def bo_sizes(role, nh):
    """Byte size of every BO a role allocates: what memlock has to admit."""
    out = []
    for k in ("a", "b", "c"):
        o = role.get(k)
        if not isinstance(o, dict) or "rows" not in o or "into" in o or "from" in o:
            continue                    # views into another role's buffer
        if o.get("stack") == "per head":
            out.append(nh * o["head_stride_bytes"])
        else:
            out.append(o["rows"] * o["cols"] * (4 if o["layout"].endswith("f32") else 2))
    if "buffer_bytes" in role:          # softmax in/out: the attention stack
        out.append(role["buffer_bytes"])
    elif "rows" in role and "in" in role:   # rowops: in and out buffers
        el_in = 4 if role["in"] == "f32" else 2
        out.append(role["rows"] * role["cols"] * el_in)
        out.append(role["rows"] * role.get("out_cols", role["cols"]) * 2)
    return out


def describe(name, kind):
    """Shape fields parsed back from a design name, for the designs table."""
    parts = name.split("_")
    if kind == "gemm":
        M, K, N = map(int, parts[1].split("x"))
        return {"kind": "gemm", "M": M, "K": K, "N": N,
                "tiles": [int(x) for x in parts[2:5]], "dtype_in": "bf16",
                "dtype_out": "bf16" if parts[0] == "mmbf" else "f32"}
    variant = parts[0]
    dims = parts[1]
    cores = int(dims.split("c")[-1])
    dims = dims.rsplit("c", 1)[0]
    valid = int(dims.split("v")[1]) if "v" in dims else 0
    dims = dims.split("v")[0]
    out_cols = int(dims.split("o")[1]) if "o" in dims else 0
    rows, cols = map(int, dims.split("o")[0].split("x"))
    d = {"kind": kind, "variant": variant, "rows": rows, "cols": cols, "cores": cores}
    if valid:
        d["valid_cols"] = valid
    if out_cols:
        d["out_cols"] = out_cols
    d["dtype_in"] = "f32" if variant == "g4f" else "bf16"
    d["dtype_out"] = "bf16"
    return d


def toolchain(vcs_ref):
    ironpy = os.path.join(rawxrt.MLIR_AIE, "ironenv/bin/python")
    v = subprocess.run([ironpy, "-c", "import importlib.metadata as m;"
                        "print(m.version('mlir_aie'), m.version('llvm-aie'))"],
                       capture_output=True, text=True).stdout.split()
    xrt = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "libxrt-npu2"],
                         capture_output=True, text=True).stdout.strip()
    return {"mlir_aie": v[0] if v else None, "peano": v[1] if len(v) > 1 else None,
            "xrt_built": xrt or None, "whisper_xdna": vcs_ref}


def device_info():
    def rd(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return None
    devs = sorted(Path("/sys/bus/pci/drivers/amdxdna").glob("0000:*")) \
        if Path("/sys/bus/pci/drivers/amdxdna").exists() else []
    dev = devs[0] if devs else None
    ids = None
    if dev:
        ids = f"{rd(dev / 'vendor')[2:]}:{rd(dev / 'device')[2:]}"
    return {"generation": "xdna1", "pci": [ids] if ids else [],
            "vbnv": rd(dev / "vbnv") if dev else None,
            "max_hw_contexts": rawxrt.MAX_CTX,
            "firmware_tested": [rd(dev / "fw_version")] if dev else [],
            "xrt_min": "2.21.75",
            "xrt_tested": ["2.21.75", "2.25.0"],
            "driver": {"in_tree": True, "kernel_release": os.uname().release,
                       "srcversion": rd("/sys/module/amdxdna/srcversion")}}


def verify(W, hp, ladder, roles, fixtures_mel=None, fixtures_dir=None):
    """One real pass per step; the designs it opens must be exactly the roles'."""
    import npu_whisper_encoder as E
    touched = []
    groups = {}             # kind -> {arg role -> set of kernel.group_id() values}
    orig = rawxrt.RawRT._ctx_entry

    def spy(self, name):
        touched.append(name)
        e = orig(self, name)
        kind = "gemm" if name.startswith("mm") else "rowop"
        for role, idx in KERNEL_ARGS[kind].items():
            groups.setdefault(kind, {}).setdefault(role, set()).add(int(e.kernel.group_id(idx)))
        return e

    rawxrt.RawRT._ctx_entry = spy
    rt = rawxrt.RawRT()
    W = dict(W)
    for k in [k for k in list(W) if k.endswith(".weight") and W[k].ndim == 2
              and not k.endswith("_ln.weight")]:
        W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)
    d = hp["n_audio_state"]
    extra = E.prepare_mlp_chunks(W, hp["n_audio_layer"], d, 0)
    rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)] + extra)
    E.gelu = rt.gelu
    kw = dict(prof={}, npu_softmax=False, n_layer=hp["n_audio_layer"],
              n_head=hp["n_audio_head"], attn_fn=rt.attn_chain, mlp_fn=rt.mlp,
              fold_bias=True)
    report = {}
    for ctx in ladder:
        # release the previous geometry the way serve/ does on a switch
        rt._chains.clear(); rt._scratch.clear(); rt._scratch_owner.clear()
        while rt._live:
            rt._evict_lru()
        rt._views.clear()
        mel = np.zeros((hp["n_mels"], 2 * ctx), np.float32) if fixtures_mel is None \
            else np.ascontiguousarray(fixtures_mel[:, :2 * ctx])
        del touched[:]
        taps = {} if (fixtures_dir and ctx == ladder[0]) else None
        out = E.encoder(W, mel, rt.matmul, rt.matmul, taps=taps, **kw)
        want = {r["design"] for r in roles[str(ctx)].values()}
        got = set(touched)
        ok = got == want and not rt.fallbacks
        report[ctx] = {"ok": ok, "missing": sorted(want - got), "extra": sorted(got - want),
                       "fallbacks": list(rt.fallbacks)}
        print(f"verify ctx={ctx}: {'ok' if ok else 'MISMATCH'} {report[ctx]}", flush=True)
        if taps is not None:
            os.makedirs(fixtures_dir, exist_ok=True)
            np.save(os.path.join(fixtures_dir, "mel.npy"), mel)
            for k in ("conv1_gelu", "block0_in", "block1_in"):
                np.save(os.path.join(fixtures_dir, f"{k}.npy"), taps[k])
            np.save(os.path.join(fixtures_dir, "out.npy"), out.astype(np.float32))
    rawxrt.RawRT._ctx_entry = orig
    # group_id() is (per-context value << 16) | bank: the high half changes with
    # every hw_context opened, so only the bank is a property of the pack
    observed = {k: {r: {"low16": sorted({x & 0xFFFF for x in v}),
                        "distinct_values": len(v)} for r, v in roles_.items()}
                for k, roles_ in groups.items()}
    return report, observed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ggml", required=True, help="whisper.cpp model file the pack is for")
    ap.add_argument("--name", required=True, help="model name, e.g. large-v3-turbo")
    ap.add_argument("--ladder", default=",".join(map(str, DEFAULT_LADDER)))
    ap.add_argument("--out", required=True, help="the pack goes in <out>/<pack id>/")
    ap.add_argument("--vcs-ref", default=os.environ.get("VCS_REF", "unknown"))
    ap.add_argument("--fixtures", help="WAV for per-stage golden dumps at the first step")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    ladder = [int(x) for x in a.ladder.split(",")]
    W, hp, types = load_encoder(a.ggml, with_types=True)
    bad = sorted({t for t in types.values() if t not in WEIGHT_TYPES + ["f32"]})
    if bad:
        sys.exit(f"{a.ggml}: weight types {bad} not supported (want {WEIGHT_TYPES})")
    if max(ladder) > hp["n_audio_ctx"]:
        sys.exit(f"ladder {ladder} exceeds n_audio_ctx {hp['n_audio_ctx']}")

    roles, kinds, attn, mlps = {}, {}, {}, {}
    for ctx in ladder:
        r, k, at, mlp = roles_for(ctx, hp)
        roles[str(ctx)], attn[ctx], mlps[ctx] = r, at, mlp
        kinds.update(k)

    designs = {}
    for name, kind in sorted(kinds.items()):
        designs[name] = describe(name, kind)
        for ext, key in (("xclbin", "xclbin"), ("bin", "insts")):
            src = os.path.join(rawxrt.AOT, f"{name}.{ext}")
            designs[name][key] = {"path": f"designs/{name}.{'insts' if ext == 'bin' else ext}",
                                  "bytes": os.path.getsize(src), "sha256": sha256(src)}
        designs[name]["kernel"] = "MLIR_AIE"

    content = hashlib.sha256("".join(
        designs[n]["xclbin"]["sha256"] + designs[n]["insts"]["sha256"]
        for n in sorted(designs)).encode()).hexdigest()
    pack_id = f"{a.name}-xdna1-{content[:12]}"

    max_bo = 0
    for ctx in ladder:
        for role in roles[str(ctx)].values():
            max_bo = max([max_bo] + bo_sizes(role, hp["n_audio_head"]))
    floor = max(HEAP_BYTES, max_bo)
    memlock = 1 << (floor - 1).bit_length()

    ladder_out = []
    for ctx in ladder:
        s = (ctx - SELECTION["margin_positions"]) / (
            SELECTION["positions_per_sec"] * (1 + SELECTION["margin_ratio"]))
        ladder_out.append({"ctx": ctx, "max_audio_secs": 30.0 if ctx == hp["n_audio_ctx"]
                           else round(s, 3), "attn": attn[ctx], "mlp": mlps[ctx],
                           "roles": roles[str(ctx)]})

    manifest = {
        "schema": SCHEMA,
        "pack_id": pack_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"name": a.name, **{k: hp[k] for k in (
                      "n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer")},
                  "weights_from": "ggml", "weight_types": WEIGHT_TYPES,
                  "validated_with": {"file": os.path.basename(a.ggml), "sha256": sha256(a.ggml)}},
        "device": device_info(),
        "toolchain": toolchain(a.vcs_ref),
        "memory": {"heap_bytes": HEAP_BYTES, "max_bo_bytes": max_bo,
                   "memlock_min_bytes": memlock,
                   "memlock_rule": "next power of two >= max(heap_bytes, max_bo_bytes)"},
        "abi": {"opcode": 3, "kernel_args": KERNEL_ARGS,
                "memory_groups": "query kernel.group_id(arg_index) on the kernel a BO feeds, "
                                 "at run time: the value changes with every hw_context "
                                 "(high 16 bits); memory_groups_observed records only the "
                                 "low 16 bits seen at export, for reference"},
        "encoder_output": {"rows": "ctx (padding stripped)", "cols": hp["n_audio_state"],
                           "dtype": "f32", "space": "after ln_post"},
        "selection": dict(SELECTION, rule="smallest ladder ctx >= ceil(secs * positions_per_sec "
                          "* (1 + margin_ratio)) + margin_positions; the full ctx from "
                          "n_audio_ctx / positions_per_sec seconds"),
        "ladder": ladder_out,
        "designs": designs,
    }

    pack = Path(a.out) / pack_id
    fixtures_dir = str(pack / "fixtures" / f"ctx{ladder[0]}") if a.fixtures else None
    if not a.no_verify:
        mel = None
        if a.fixtures:
            import torch
            import whisper
            from audio_io import read_wav
            audio = read_wav(a.fixtures)
            mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)),
                                              n_mels=hp["n_mels"]).numpy().astype(np.float32)
        rep, observed = verify(W, hp, ladder, roles, mel, fixtures_dir)
        manifest["abi"]["memory_groups_observed"] = observed
        manifest["verified"] = {"date": manifest["created"],
                                "designs_match_ladder": all(r["ok"] for r in rep.values())}
        if not manifest["verified"]["designs_match_ladder"]:
            sys.exit(f"verification failed: {rep}")
    (pack / "designs").mkdir(parents=True, exist_ok=True)
    for name in designs:
        shutil.copyfile(os.path.join(rawxrt.AOT, f"{name}.xclbin"), pack / designs[name]["xclbin"]["path"])
        shutil.copyfile(os.path.join(rawxrt.AOT, f"{name}.bin"), pack / designs[name]["insts"]["path"])
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(pack)


if __name__ == "__main__":
    main()
