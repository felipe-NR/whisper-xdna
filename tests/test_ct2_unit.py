#!/usr/bin/env python3
"""Unit tests for serve/ct2_decoder.py, on CPU, with whisper-tiny.

Fast (a ~75 MB checkpoint, conversions of a second or two) so it can run on
every change; tests/test_ct2_decoder.py is the slow large-v3-turbo equivalence
run. What each test pins down:

  1 suppression set     suppress_ids is exactly what whisper.decode computes
                        for suppress_tokens="-1" (DecodingTask itself, not a copy)
  2 blank suppression   suppress_ids_begin is SuppressBlank's set
  3 vocabulary          one unique string per id, specials where the tokenizer
                        puts them, size equal to the checkpoint's n_vocab
  4 conversion is pure  converting leaves every tensor of the model untouched
  5 cache naming        the cache key carries the checkpoint SHA-256 that
                        openai-whisper verifies, the quantisation and revision
  6 cache hit           a cached model is returned without starting a child
  7 cache miss          a missing one is converted in a child, atomically, with
                        a manifest naming the checkpoint digest
  8 max_length          Whisper.generate returns max_length // 2 tokens; the
                        decoder relies on it to stop where whisper.decode stops
  9 same text           float32: CT2Decoder.decode == whisper.decode exactly,
                        full and short ctx; int8: the same words (tiny can flip
                        a comma; exact int8 parity is test_ct2_decoder.py's)
 10 decoder knob        the server refuses an unknown NPU_STT_DECODER
 11 language detection  CT2Decoder.language_probs picks what whisper's own
                        detect_language picks, with close probabilities
 12 ctx from audio      the server's ctx_from_audio always covers the audio
                        (50 positions/s), adds its margin, and is 1500 from 30 s

  python3 tests/test_ct2_unit.py            # refenv + ctranslate2 (ct2 image)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))

import numpy as np                 # noqa: E402
import torch                       # noqa: E402
import whisper                     # noqa: E402
from whisper.decoding import DecodingOptions, DecodingTask   # noqa: E402
from whisper.tokenizer import get_tokenizer                  # noqa: E402

import ct2_decoder                 # noqa: E402

MODEL = os.environ.get("CT2_TEST_MODEL", "tiny")
JFK = str(ROOT / "tests" / "data" / "jfk.wav")
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return ok


def features(model, mel, ctx):
    """What the NPU encoder hands the decoder for audio_ctx=ctx."""
    enc = model.encoder
    pos = enc.positional_embedding
    enc.positional_embedding = pos[:ctx]
    try:
        return enc(mel[:, :2 * ctx].unsqueeze(0))
    finally:
        enc.positional_embedding = pos


def main():
    torch.set_grad_enabled(False)
    model = whisper.load_model(MODEL, device="cpu")
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                        language="en", task="transcribe")
    opts = DecodingOptions(fp16=False, language="en", without_timestamps=True,
                           temperature=0.0)
    cache = tempfile.mkdtemp(prefix="ct2-unit-")

    print("1-3 conversion config")
    task = DecodingTask(model, opts)
    want = sorted(task._get_suppress_tokens())
    got = ct2_decoder.suppress_ids(tok)
    check("1 suppress_ids == DecodingTask._get_suppress_tokens()", got == want,
          f"{len(got)} ids" if got == want else
          f"only ours {sorted(set(got) - set(want))[:8]}, only whisper's "
          f"{sorted(set(want) - set(got))[:8]}")
    begin = tok.encode(" ") + [tok.eot]
    out_dir = ct2_decoder.convert(model, tok, MODEL, cache, log=lambda m: None)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    check("2 suppress_ids_begin == SuppressBlank's set",
          cfg["suppress_ids_begin"] == begin, f"{cfg['suppress_ids_begin']}")
    check("1b suppress_ids written to config", cfg["suppress_ids"] == want)
    vocab = json.load(open(os.path.join(out_dir, "vocabulary.json")))
    specials = {"<|endoftext|>": tok.eot, "<|startoftranscript|>": tok.sot,
                "<|notimestamps|>": tok.no_timestamps, "<|transcribe|>": tok.transcribe,
                "<|en|>": tok.to_language_token("en"), "<|pt|>": tok.to_language_token("pt")}
    if tok.no_speech is not None:
        specials["<|nospeech|>"] = tok.no_speech
    misplaced = {s: i for s, i in specials.items() if vocab[i] != s}
    check("3 vocabulary: size, uniqueness, specials in place",
          len(vocab) == model.dims.n_vocab and len(set(vocab)) == len(vocab)
          and not misplaced,
          f"{len(vocab)} ids" + (f", misplaced {misplaced}" if misplaced else ""))
    check("3b lang_ids are the tokenizer's language tokens",
          cfg["lang_ids"] == list(tok.all_language_tokens))

    print("4 conversion leaves the model untouched")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ct2_decoder.convert(model, tok, MODEL, tempfile.mkdtemp(prefix="ct2-unit-"),
                        log=lambda m: None)
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    check("4 state_dict identical after convert()", not changed,
          f"{len(changed)} changed, e.g. {changed[:3]}" if changed else
          f"{len(before)} tensors")
    del before

    print("5-7 cache")
    sha = ct2_decoder.checkpoint_sha256(MODEL)
    url_sha = whisper._MODELS[MODEL].split("/")[-2]
    p8, _ = ct2_decoder.cache_path(MODEL, cache, "int8")
    pf, _ = ct2_decoder.cache_path(MODEL, cache, "float32")
    check("5 cache key: checkpoint digest, quantisation, revision",
          sha == url_sha and sha[:16] in p8 and p8.endswith(
              f"-int8-r{ct2_decoder.CONVERTER_REVISION}") and p8 != pf,
          os.path.basename(p8))
    real_run = subprocess.run
    try:
        def no_child(*a, **k):
            raise AssertionError("ensure_model started a child on a cache hit")
        subprocess.run = no_child
        hit = ct2_decoder.ensure_model(MODEL, cache, "int8", log=lambda m: None)
        check("6 cache hit returns the cached dir, no child", hit == out_dir)
    except AssertionError as e:
        check("6 cache hit returns the cached dir, no child", False, str(e))
    finally:
        subprocess.run = real_run
    fresh = tempfile.mkdtemp(prefix="ct2-unit-")
    calls = []

    def spy(cmd, *a, **k):
        calls.append(cmd)
        return real_run(cmd, *a, **k)
    subprocess.run = spy
    try:
        miss = ct2_decoder.ensure_model(MODEL, fresh, "int8", log=lambda m: None)
    finally:
        subprocess.run = real_run
    man = json.load(open(os.path.join(miss, "manifest.json")))
    leftovers = [n for n in os.listdir(fresh) if ".tmp-" in n]
    check("7 cache miss converts in a child process",
          len(calls) == 1 and calls[0][0] == sys.executable
          and calls[0][2].endswith("ct2_decoder.py"), f"{len(calls)} child(ren)")
    check("7b manifest names the checkpoint digest and quantisation",
          man["checkpoint_sha256"] == sha and man["quantization"] == "int8"
          and man["converter_revision"] == ct2_decoder.CONVERTER_REVISION)
    check("7c no temporary directory left behind", not leftovers, f"{leftovers}")

    print("8-9 decoding")
    import ctranslate2
    dec = ct2_decoder.CT2Decoder(out_dir, compute_type="int8", threads=4)
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(JFK)),
                                      n_mels=model.dims.n_mels)
    full = features(model, mel, 1500)
    x = ctranslate2.StorageView.from_array(np.ascontiguousarray(full.numpy(), dtype=np.float32))
    prompt = list(tok.sot_sequence_including_notimestamps)
    lens = {ml: len(dec.model.generate(x, [prompt], beam_size=1, max_length=ml)[0]
                    .sequences_ids[0]) for ml in (10, 20)}
    check("8 Whisper.generate returns max_length // 2 tokens",
          lens == {10: 5, 20: 10}, f"{lens}")

    class Stub(torch.nn.Module):
        feats = None

        def forward(self, _mel):
            return self.feats
    # Two questions, kept apart. Is the conversion (weights, config, decoding
    # options) right? Exact text, in float32, where nothing but the conversion
    # can make the decoders differ. Does int8 keep the content? Same words: on
    # whisper-tiny int8 can flip a near-tie such as a comma (measured: "for
    # you, ask" vs "for you ask" at ctx=1500). Exact int8 parity is required
    # where it is claimed, on large-v3-turbo, by tests/test_ct2_decoder.py.
    f32_dir = ct2_decoder.convert(model, tok, MODEL, cache, quantization="float32",
                                  log=lambda m: None)
    dec32 = ct2_decoder.CT2Decoder(f32_dir, compute_type="float32", threads=4)
    norm = lambda s: re.sub(r"[^\w\s]", " ", s.lower()).split()
    enc, stub = model.encoder, Stub()
    for ctx in (1500, 768):
        feats = full if ctx == 1500 else features(model, mel, ctx)
        model.encoder = stub
        stub.feats = feats
        try:
            ref = whisper.decode(model, mel.unsqueeze(0), opts)[0].text.strip()
        finally:
            model.encoder = enc
        got32 = dec32.decode(feats[0].numpy(), tok)
        check(f"9 float32 decode == whisper.decode at ctx={ctx}", got32 == ref,
              repr(got32) if got32 == ref else f"ct2 {got32!r} vs torch {ref!r}")
        got8 = dec.decode(feats[0].numpy(), tok)
        check(f"9b int8 decode: same words as whisper.decode at ctx={ctx}",
              norm(got8) == norm(ref),
              "identical" if got8 == ref else f"punctuation differs: {got8!r}")
        check(f"9c decode returns stripped text at ctx={ctx}", got8 == got8.strip())

    print("10 server configuration")
    r = subprocess.run([sys.executable, str(ROOT / "serve" / "npu_stt_server.py")],
                       env={**os.environ, "NPU_STT_DECODER": "bogus"},
                       capture_output=True, text=True, timeout=120)
    check("10 NPU_STT_DECODER=bogus is refused at start",
          r.returncode != 0 and "expected torch or ct2" in (r.stderr + r.stdout),
          f"exit {r.returncode}")

    print("11 language detection")
    probs = dec.language_probs(full[0].numpy())
    _, tp = whisper.detect_language(model, full, tok)
    tp = tp[0]
    top_ct2, top_torch = max(probs, key=probs.get), max(tp, key=tp.get)
    check("11 language_probs covers every language token",
          set(probs) == set(tp), f"{len(probs)} vs {len(tp)}")
    check("11b same top language as whisper.detect_language on jfk.wav",
          top_ct2 == top_torch == "en",
          f"ct2 {top_ct2} {probs[top_ct2]:.3f}, torch {top_torch} {tp[top_torch]:.3f}")
    check("11c probabilities close (int8 vs fp32)",
          abs(probs["en"] - tp["en"]) < 0.05, f"|delta| {abs(probs['en'] - tp['en']):.4f}")

    print("12 ctx from audio")
    import importlib.util
    spec = importlib.util.spec_from_file_location("srv", ROOT / "serve" / "npu_stt_server.py")
    srv = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(srv)       # module level only; main() is not run
    finally:
        os.chdir(cwd)
    sr = srv.SR
    cases = {0.5: 44, 5.95: 344, 11.0: 621, 29.9: 1500, 30.0: 1500, 45.0: 1500}
    got = {s: srv.ctx_from_audio(int(s * sr)) for s in cases}
    check("12 ctx_from_audio: 50/s with a 10% + 16 margin, 1500 from 30 s",
          got == cases, f"{got}")
    covers = all(srv.ctx_for(srv.ctx_from_audio(int(s * sr))) >= min(1500, s * 50)
                 for s in (0.1, 1, 3, 7.3, 10.2, 13, 19.9, 25.5, 29.99, 31, 60))
    check("12b rounded onto the ladder it never covers less than the audio", covers)

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
