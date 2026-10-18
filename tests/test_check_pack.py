#!/usr/bin/env python3
"""tools/check_pack.py against the committed pack manifests. Host, stdlib only.

  1 committed manifests  every tests/data/packs/*/manifest.json validates
                         (manifest only: the overlays are not versioned)
  2 refusals             a copy with one defect each must be refused:
                         unknown schema major, unknown variant, a path out of
                         designs/, memlock off its rule, a selection figure
                         off its rule, a role naming a missing design,
                         a pack id that is not the hash of its designs
  3 fixtures             the tiny pack's golden dumps exist with the shapes
                         docs/pack-format.md promises
"""
import copy
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = [sys.executable, "-I", str(ROOT / "tools" / "check_pack.py")]
PACKS = sorted((ROOT / "tests" / "data" / "packs").glob("*/manifest.json"))
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def run(pack_dir):
    r = subprocess.run(CHECK + [str(pack_dir), "--no-files"], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def npy_shape(path):
    with open(path, "rb") as f:
        f.read(8)
        (hlen,) = struct.unpack("<H", f.read(2))
        hdr = f.read(hlen).decode()
    inner = hdr.split("'shape': (")[1].split(")")[0]
    return tuple(int(x) for x in inner.split(",") if x.strip())


def main():
    print("=== 1. committed manifests")
    check("there are committed packs", len(PACKS) >= 2, f"{[p.parent.name for p in PACKS]}")
    for m in PACKS:
        rc, out = run(m.parent)
        check(f"{m.parent.name} validates", rc == 0, out.splitlines()[-1] if out else "")

    print("=== 2. refusals")
    base = json.loads(PACKS[0].read_text())
    first = next(iter(base["designs"]))

    def schema(m): m["schema"] = "xdna-whisper-pack/2"
    def variant(m):
        d = next(d for d in m["designs"].values() if d["kind"] != "gemm")
        d["variant"] = "s9"
    def path(m): m["designs"][first]["xclbin"]["path"] = "designs/../x.xclbin"
    def memlock(m): m["memory"]["memlock_min_bytes"] *= 2
    def selection(m): m["ladder"][0]["max_audio_secs"] += 1
    def role(m): m["ladder"][0]["roles"]["conv1"]["design"] = "mm_nope"
    def pack_id(m): m["pack_id"] = m["pack_id"][:-1] + ("0" if m["pack_id"][-1] != "0" else "1")

    for name, mutate in (("unknown schema major", schema), ("unknown variant", variant),
                         ("path out of designs/", path), ("memlock off its rule", memlock),
                         ("selection off its rule", selection), ("missing design", role),
                         ("pack id not the content hash", pack_id)):
        m = copy.deepcopy(base)
        mutate(m)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.json").write_text(json.dumps(m))
            rc, out = run(d)
        check(f"refuses: {name}", rc != 0, out.splitlines()[-1] if out else "")

    print("=== 3. fixtures")
    fx = sorted((ROOT / "tests" / "data" / "packs").glob("tiny-*/fixtures/ctx512"))
    check("tiny fixtures present", len(fx) == 1)
    if fx:
        want = {"mel.npy": (80, 1024), "conv1_gelu.npy": (1024, 384), "block0_in.npy": (512, 384),
                "block1_in.npy": (512, 384), "out.npy": (512, 384)}
        for f, shape in want.items():
            p = fx[0] / f
            got = npy_shape(p) if p.exists() else None
            check(f"{f} {shape}", got == shape, f"got {got}")

    bad = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
