#!/usr/bin/env python3
"""Integration tests against a running STT service (serve/), any decoder.

Needs only the standard library, so it runs on the host against the published
port. What it checks:

  1 health        /health is "ok", not degraded, and names its decoder
                  (--expect-decoder makes the backend part of the contract)
  2 inference     POST /inference transcribes tests/data/jfk.wav to the expected
                  sentence, on the NPU path, with the X-NPU-* timing headers
  3 openai path   POST /v1/audio/transcriptions answers the same text
  4 short ctx     an audio_ctx that covers the clip gives the same text
  5 errors        unknown path 404, non-multipart 400, missing file 400,
                  undecodable audio 400 -- and the service is still healthy
  6 clips         optional --clips DIR: every DIR/*.wav with a .txt beside it
                  must stay within --max-wer of it
  7 A/B           --save FILE records every transcript; --against FILE fails
                  on any transcript that differs from a recorded run (how the
                  torch and ct2 decoders are compared: the NPU serves one
                  process at a time, so the two runs are sequential)

  python3 tests/test_serve_http.py --expect-decoder ct2 --save /tmp/ct2.json
"""
import argparse
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JFK = ROOT / "tests" / "data" / "jfk.wav"
# Compared after normalisation: large-v3-turbo punctuates this sentence and
# whisper-tiny does not, and neither is wrong.
EXPECTED = ("And so my fellow Americans ask not what your country can do for "
            "you ask what you can do for your country.")
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return ok


def words(text):
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s]", " ", text).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / max(1, len(r))


def request(url, path, fields=None, wav=None, raw=None, ctype=None, timeout=300):
    """POST multipart (fields + optional file) or raw bytes; (status, headers, json)."""
    if raw is None:
        b = uuid.uuid4().hex
        parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in (fields or {}).items()]
        if wav is not None:
            name, data = wav
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; '
                         f'filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                         + data + b"\r\n")
        parts.append(f"--{b}--\r\n".encode())
        raw, ctype = b"".join(parts), f"multipart/form-data; boundary={b}"
    req = urllib.request.Request(url.rstrip("/") + path, data=raw,
                                 headers={"Content-Type": ctype or "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, e.headers, json.loads(body or b"{}")
        except ValueError:
            return e.code, e.headers, {"raw": body[:200].decode(errors="replace")}


def health(url):
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--expect-decoder", choices=["torch", "ct2"])
    ap.add_argument("--clips", type=Path, help="dir of .wav + .txt pairs")
    ap.add_argument("--language", default="pt", help="language for --clips")
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--against", type=Path)
    a = ap.parse_args()
    transcripts = {}

    print("1 health")
    h = health(a.url)
    dec = h.get("decoder") or {}
    check("1 status ok and not degraded", h.get("status") == "ok" and not h.get("degraded"),
          f"status={h.get('status')} degraded={h.get('degraded')}")
    check("1b health names the decoder", dec.get("backend") in ("torch", "ct2"), f"{dec}")
    if a.expect_decoder:
        check(f"1c decoder is {a.expect_decoder}", dec.get("backend") == a.expect_decoder)

    print("2-4 transcription")
    jfk = (JFK.name, JFK.read_bytes())
    st, hd, body = request(a.url, "/inference",
                           {"language": "en", "audio_ctx": "1500", "response_format": "json"}, jfk)
    text = body.get("text", "")
    transcripts["jfk@1500/inference"] = text
    check("2 /inference transcribes jfk.wav", st == 200 and words(text) == words(EXPECTED),
          repr(text) if st == 200 else f"HTTP {st} {body}")
    check("2b served by the NPU", hd.get("X-STT-Path") == "npu", f"X-STT-Path={hd.get('X-STT-Path')}")
    timing = {k: hd.get(f"X-NPU-{k}-Ms") for k in ("Enc", "Dec", "Total")}
    check("2c X-NPU-* timing headers present and sane",
          all(v is not None for v in timing.values())
          and float(timing["Total"]) >= float(timing["Enc"]) + float(timing["Dec"]) - 1,
          f"{timing}")
    st2, _, body2 = request(a.url, "/v1/audio/transcriptions",
                            {"language": "en", "audio_ctx": "1500", "model": "whisper-1"}, jfk)
    transcripts["jfk@1500/openai"] = body2.get("text", "")
    check("3 /v1/audio/transcriptions answers the same text",
          st2 == 200 and body2.get("text") == text, repr(body2.get("text")))
    # jfk.wav is 11.0 s: 550 positions at 50/s, which the ladder rounds to 768.
    st3, hd3, body3 = request(a.url, "/inference", {"language": "en", "audio_ctx": "600"}, jfk)
    transcripts["jfk@600/inference"] = body3.get("text", "")
    check("4 short audio_ctx covering the clip: same words",
          st3 == 200 and words(body3.get("text", "")) == words(EXPECTED),
          f"ctx={hd3.get('X-NPU-Ctx')} {body3.get('text')!r}")

    print("5 errors")
    st, _, _ = request(a.url, "/nope", {"language": "en"}, jfk)
    check("5 unknown path -> 404", st == 404, f"HTTP {st}")
    st, _, _ = request(a.url, "/inference", raw=b"not multipart", ctype="text/plain")
    check("5b non-multipart -> 400", st == 400, f"HTTP {st}")
    st, _, _ = request(a.url, "/inference", {"language": "en"})
    check("5c missing file -> 400", st == 400, f"HTTP {st}")
    st, _, _ = request(a.url, "/inference", {"language": "en"}, ("x.wav", b"\x00garbage" * 64))
    check("5d undecodable audio -> 400", st == 400, f"HTTP {st}")
    h2 = health(a.url)
    check("5e still healthy after the errors", h2.get("status") == "ok" and not h2.get("degraded"))

    if a.clips:
        print("6 clips")
        for wav in sorted(a.clips.glob("*.wav")):
            ref = wav.with_suffix(".txt")
            if not ref.exists():
                continue
            st, hd, body = request(a.url, "/inference", {"language": a.language},
                                   (wav.name, wav.read_bytes()))
            got = body.get("text", "")
            transcripts[f"{wav.name}/inference"] = got
            e = wer(ref.read_text(encoding="utf-8"), got)
            check(f"6 {wav.name}: WER {e:.1%} <= {a.max_wer:.0%}", st == 200 and e <= a.max_wer,
                  f"dec {hd.get('X-NPU-Dec-Ms')} ms")

    if a.save:
        a.save.write_text(json.dumps({"decoder": dec, "transcripts": transcripts},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved {len(transcripts)} transcripts to {a.save}")
    if a.against:
        print("7 A/B against a recorded run")
        prev = json.loads(a.against.read_text(encoding="utf-8"))
        for k, v in prev["transcripts"].items():
            if k in transcripts:
                check(f"7 {k} identical to the {prev['decoder'].get('backend')} run",
                      transcripts[k] == v,
                      "" if transcripts[k] == v else f"{transcripts[k]!r} vs {v!r}")

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
