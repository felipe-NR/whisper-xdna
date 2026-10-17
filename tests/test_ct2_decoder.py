#!/usr/bin/env python3
"""The CTranslate2 decoder must say what the torch decoder says.

CPU only, no NPU: encoder output comes from the torch reference encoder, so
this isolates the decoder swap. For every clip and every context it checks
that serve/ct2_decoder.py, converted from the checkpoint whisper.load_model
returns, produces the same text as whisper.decode with serve/'s options.
Contexts below 1500 trim the encoder output, which is what a request with a
small audio_ctx hands the decoder.

  python3 tests/test_ct2_decoder.py [--reference DIR] [clip.wav ...]

--reference points at a model converted by CTranslate2's own Transformers
converter (e.g. from openai/whisper-large-v3-turbo); its vocabulary and
suppression config must then match this converter's, token for token.

Needs torch, openai-whisper and ctranslate2: refenv plus ctranslate2.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))

import torch                      # noqa: E402
import whisper                    # noqa: E402
from whisper.tokenizer import get_tokenizer   # noqa: E402

import ct2_decoder                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", default=[str(ROOT / "tests/data/jfk.wav")])
    ap.add_argument("--model", default=os.environ.get("NPU_STT_MODEL", "large-v3-turbo"))
    ap.add_argument("--language", default=None,
                    help="default: per clip, en for jfk.wav, else pt")
    ap.add_argument("--ctx", default="1500,512")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--cache", default=None, help="keep the converted model here")
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    bad = []

    torch.set_grad_enabled(False)
    model = whisper.load_model(a.model, device="cpu")
    tok_for = lambda lang: get_tokenizer(model.is_multilingual,
                                         num_languages=model.num_languages,
                                         language=lang, task="transcribe")
    cache = a.cache or tempfile.mkdtemp(prefix="ct2-test-")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # In-process on purpose (serve/ converts in a child): this is the call that
    # must leave the model it reads untouched.
    out = ct2_decoder.convert(model, tok_for("en"), a.model, cache,
                              quantization=a.compute_type.split("_")[0])
    # Conversion must not touch the model it reads: in serve/ the encoder
    # weights and (on the torch path) the decoder come out of the same object.
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    print(f"model weights untouched by conversion: {not changed}"
          + (f" ({len(changed)} changed, e.g. {changed[:3]})" if changed else ""))
    if changed:
        bad.append("conversion mutated the model")
    del before
    dec = ct2_decoder.CT2Decoder(out, compute_type=a.compute_type, threads=8)

    class _Stub(torch.nn.Module):
        feats = None

        def forward(self, _mel):
            return self.feats

    encoder, stub = model.encoder, _Stub()
    model.encoder = stub

    if a.reference:
        ref_vocab = json.load(open(os.path.join(a.reference, "vocabulary.json")))
        own_vocab = json.load(open(os.path.join(out, "vocabulary.json")))
        same = ref_vocab == own_vocab
        print(f"vocabulary vs reference: {len(own_vocab)} ids, identical={same}")
        if not same:
            diff = [i for i, (x, y) in enumerate(zip(ref_vocab, own_vocab)) if x != y]
            print(f"  first differing ids: {diff[:10]} (len {len(ref_vocab)} vs {len(own_vocab)})")
            bad.append("vocabulary")
        rc = json.load(open(os.path.join(a.reference, "config.json")))
        oc = json.load(open(os.path.join(out, "config.json")))
        for k in ("suppress_ids", "suppress_ids_begin", "lang_ids"):
            same = sorted(rc.get(k) or []) == sorted(oc.get(k) or [])
            print(f"config {k} vs reference: identical={same}")
            if not same:
                print(f"  only reference: {sorted(set(rc.get(k) or []) - set(oc.get(k) or []))[:20]}")
                print(f"  only ours:      {sorted(set(oc.get(k) or []) - set(rc.get(k) or []))[:20]}")
                bad.append(k)

    for w in a.wavs:
        lang = a.language or ("en" if w.endswith("jfk.wav") else "pt")
        tok = tok_for(lang)
        opts = whisper.DecodingOptions(fp16=False, language=lang,
                                       without_timestamps=True, beam_size=None,
                                       temperature=0.0)
        audio = whisper.pad_or_trim(whisper.load_audio(w))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        pos = encoder.positional_embedding
        for ctx in (int(c) for c in a.ctx.split(",")):
            # What the NPU encoder does for audio_ctx=ctx (and whisper.cpp -ac):
            # encode only the first 2*ctx mel frames against the first ctx
            # positions -- not a 1500-position encoding cut short afterwards.
            encoder.positional_embedding = pos[:ctx]
            feats = encoder(mel[:, :2 * ctx].unsqueeze(0))
            encoder.positional_embedding = pos
            # The same substitution serve/ makes: whisper.decode runs
            # model.encoder on the mel, and the stub answers with the features.
            stub.feats = feats
            t0 = time.perf_counter()
            ref = whisper.decode(model, mel.unsqueeze(0), opts)[0].text.strip()
            t_ref = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            got = dec.decode(feats[0].numpy(), tok).strip()
            t_ct2 = (time.perf_counter() - t0) * 1e3
            ok = got == ref
            print(f"{'PASS' if ok else 'FAIL'} {Path(w).name} ctx={ctx} {lang}: "
                  f"torch {t_ref:.0f} ms, ct2 {t_ct2:.0f} ms")
            if not ok:
                print(f"  torch: {ref}\n  ct2:   {got}")
                bad.append(f"{Path(w).name}@{ctx}")

    print("OK" if not bad else f"FAILED: {', '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
