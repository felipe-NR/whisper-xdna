#!/usr/bin/env python3
"""Compare dictation latency and accuracy: iGPU (voxtype/Vulkan) vs NPU (serve/).

Each WAV must be 16 kHz mono and have its reference transcript next to it with
the same basename and a .txt extension. Both paths transcribe the same file:

  gpu        `voxtype -v --model M --language L transcribe <wav>`. Only the
             inference time voxtype logs is counted; its model load is
             reported apart, since the daemon keeps the model resident.
  npu@<ctx>  POST /inference on the NPU service, once per audio_ctx variant.
             "auto" sizes the context to the clip (50 positions per second
             plus a margin) and the server rounds it up onto its ladder.

  python3 -I tools/compare_dictation.py --runs 3 audio/*.wav

Standard library only, so it runs on the host without the image's venvs.
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.request
import uuid
import wave
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DONE = re.compile(r'Transcription completed in ([0-9.]+)s: "(.*)"$')
LOADED = re.compile(r"Model loaded in ([0-9.]+)s")


def duration_s(path):
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            sys.exit(f"{path}: expected 16 kHz mono, got "
                     f"{w.getframerate()} Hz x{w.getnchannels()}")
        return w.getnframes() / w.getframerate()


def words(text):
    """Lowercase, drop punctuation, keep accents: pt-BR diacritics matter."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / max(1, len(r))


def run_gpu(wav, model, language):
    t0 = time.perf_counter()
    p = subprocess.run(
        ["voxtype", "-v", "--model", model, "--language", language,
         "transcribe", str(wav)],
        capture_output=True, text=True, check=True)
    wall = (time.perf_counter() - t0) * 1e3
    load = infer = None
    for line in (ANSI.sub("", l) for l in (p.stderr + p.stdout).splitlines()):
        if m := LOADED.search(line):
            load = float(m.group(1)) * 1e3
        if m := DONE.search(line):
            infer = float(m.group(1)) * 1e3
    if infer is None:
        raise RuntimeError(f"no completion line from voxtype:\n{p.stderr[-2000:]}")
    # The log line truncates long transcripts with "..."; the full text is the
    # last non-empty line voxtype prints on stdout.
    out = [l for l in ANSI.sub("", p.stdout).splitlines() if l.strip()]
    text = out[-1].strip() if out else ""
    return {"text": text, "infer_ms": infer, "load_ms": load, "wall_ms": wall}


def run_npu(wav, url, language, ctx):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("language", language), ("response_format", "json"),
                        ("audio_ctx", str(ctx))):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; '
                     f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{wav.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + wav.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url.rstrip("/") + "/inference", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
        h = r.headers
    wall = (time.perf_counter() - t0) * 1e3
    if h.get("X-STT-Path") != "npu":
        raise RuntimeError(f"served by {h.get('X-STT-Path')!r}, not the NPU")
    num = lambda k: float(h[k]) if h.get(k) else None
    return {"text": body.get("text", ""), "infer_ms": num("X-NPU-Total-Ms"),
            "enc_ms": num("X-NPU-Enc-Ms"), "dec_ms": num("X-NPU-Dec-Ms"),
            "mel_ms": num("X-NPU-Mel-Ms"), "ctx": int(h.get("X-NPU-Ctx", 0)),
            "wall_ms": wall}


def auto_ctx(seconds):
    return min(1500, math.ceil(seconds * 50 * 1.1) + 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="pt")
    ap.add_argument("--npu-url", default="http://127.0.0.1:8090")
    ap.add_argument("--npu-ctx", default="auto,1500",
                    help="comma-separated audio_ctx values; 'auto' fits the clip")
    ap.add_argument("--json", type=Path, help="write raw results here")
    a = ap.parse_args()

    results = []
    for wav in a.wavs:
        ref = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        dur = duration_s(wav)
        paths = [("gpu", lambda: run_gpu(wav, a.model, a.language))]
        for c in a.npu_ctx.split(","):
            ctx = auto_ctx(dur) if c == "auto" else int(c)
            paths.append((f"npu@{c}",
                          lambda ctx=ctx: run_npu(wav, a.npu_url, a.language, ctx)))
        for name, fn in paths:
            fn()  # warm-up: shader compilation, first overlay switch
            runs = [fn() for _ in range(a.runs)]
            results.append({"wav": wav.name, "duration_s": dur, "path": name,
                            "ref": ref, "runs": runs})
            print(f"{wav.name} {name}: "
                  f"{[round(r['infer_ms']) for r in runs]} ms", file=sys.stderr)

    print("| Áudio | Caminho | Inferência (mediana) | RTF | Detalhe | WER | Transcrição |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        med = statistics.median(x["infer_ms"] for x in r["runs"])
        last = r["runs"][-1]
        if r["path"] == "gpu":
            detail = f"carga do modelo {last['load_ms']:.0f} ms (fora da mediana)"
        else:
            detail = (f"ctx={last['ctx']}, enc {last['enc_ms']:.0f} + "
                      f"dec {last['dec_ms']:.0f} ms")
        print(f"| {r['wav']} ({r['duration_s']:.1f} s) | {r['path']} | "
              f"{med:.0f} ms | {med / 1e3 / r['duration_s']:.2f} | {detail} | "
              f"{wer(r['ref'], last['text']):.1%} | {last['text']} |")

    if a.json:
        a.json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                          encoding="utf-8")


if __name__ == "__main__":
    main()
