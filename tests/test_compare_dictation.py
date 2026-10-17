#!/usr/bin/env python3
"""Unit tests for tools/compare_dictation.py's pure helpers. Host, stdlib only.

  1 words         lowercases, drops punctuation, keeps pt-BR diacritics
  2 wer           edit distance over words, normalised by the reference
  3 auto_ctx      covers the clip with a margin, never past 1500
  4 voxtype log   the completion line is parsed, colours and all; the text is
                  never taken from it (voxtype truncates it with "...")
  5 duration_s    16 kHz mono accepted, anything else refused
"""
import importlib.util
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cd", ROOT / "tools" / "compare_dictation.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def wav(path, rate, channels, seconds):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * int(rate * seconds))


def main():
    print("1 words")
    check("1 punctuation dropped, accents kept",
          cd.words("Amanhã de manhã, o relatório!") == ["amanhã", "de", "manhã", "o", "relatório"])
    check("1b hyphenated and decomposed forms",
          cd.words("e-mail É") == ["e", "mail", "é"], f"{cd.words('e-mail E' + chr(0x301))}")

    print("2 wer")
    check("2 identical -> 0", cd.wer("o gato preto", "O gato, preto.") == 0)
    check("2b one substitution in four", cd.wer("o gato é preto", "o rato é preto") == 0.25)
    check("2c deletion and insertion",
          cd.wer("a b c", "a c") == 1 / 3 and cd.wer("a b", "a x b") == 0.5)
    check("2d empty hypothesis -> 1", cd.wer("a b c", "") == 1)
    check("2e empty reference does not divide by zero", cd.wer("", "x") == 1)

    print("3 auto_ctx")
    # ceil(5.95 * 50 * 1.1) + 16 = 328 + 16
    check("3 6 s clip -> 344 positions (298 needed)", cd.auto_ctx(5.95) == 344,
          f"{cd.auto_ctx(5.95)}")
    check("3b covers 50 positions per second", all(cd.auto_ctx(s) >= s * 50 for s in (0.5, 6, 11, 20, 27)))
    check("3c capped at 1500", cd.auto_ctx(29.9) == 1500 and cd.auto_ctx(60) == 1500)

    print("4 voxtype log parsing")
    line = ("\x1b[2m2026-10-17T18:40:22.771826Z\x1b[0m \x1b[32m INFO\x1b[0m "
            "Transcription completed in 1.44s: \"Amanhã de manhã preciso revisar o relatório e envi...\"")
    m = cd.DONE.search(cd.ANSI.sub("", line))
    check("4 completion line parsed through ANSI colours",
          m is not None and float(m.group(1)) == 1.44)
    m2 = cd.LOADED.search(cd.ANSI.sub("", "\x1b[32m INFO\x1b[0m Model loaded in 0.40s"))
    check("4b model load time parsed", m2 is not None and float(m2.group(1)) == 0.40)
    # What voxtype -v transcribe actually prints: logs on stderr with the text
    # cut short, the full text as the last stdout line.
    full = "Amanhã de manhã preciso revisar o relatório e enviar o e-mail para a equipe."
    fake = cd.subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=f'Loading audio file: "x.wav"\nProcessing 95205 samples (5.95s)...\n\n{full}\n',
        stderr=("\x1b[32m INFO\x1b[0m Model loaded in 0.34s\n" + line + "\n"))
    real = cd.subprocess.run
    cd.subprocess.run = lambda *a, **k: fake
    try:
        r = cd.run_gpu(Path("x.wav"), "large-v3-turbo", "pt")
    finally:
        cd.subprocess.run = real
    check("4c run_gpu: full text from stdout, not the truncated log line",
          r["text"] == full, repr(r["text"]))
    check("4d run_gpu: inference and load times from the log",
          r["infer_ms"] == 1440 and abs(r["load_ms"] - 340) < 1e-6, f"{r['infer_ms']}, {r['load_ms']}")

    print("5 duration_s")
    d = Path(tempfile.mkdtemp())
    wav(d / "ok.wav", 16000, 1, 1.5)
    check("5 16 kHz mono accepted", abs(cd.duration_s(d / "ok.wav") - 1.5) < 1e-6)
    for name, rate, ch in (("44k.wav", 44100, 1), ("stereo.wav", 16000, 2)):
        wav(d / name, rate, ch, 0.5)
        try:
            cd.duration_s(d / name)
            check(f"5b {name} refused", False)
        except SystemExit:
            check(f"5b {name} refused", True)

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
