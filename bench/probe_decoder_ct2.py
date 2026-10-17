"""CTranslate2 as the CPU decoder: same encoder output, cheaper arithmetic.

`ctranslate2.models.Whisper.generate()` accepts encoder output
([batch, 1500, d_model]) as well as a mel, which is what would let serve/ keep
the NPU encoder and swap only the decoder. The encoder output here comes from
the torch reference encoder, computed once per clip, exactly as in
probe_decoder_cpu.py, so the two probes time the same decoding problem.

  docker run --rm -v whisper-xdna_models:/root/.cache/whisper \
    -v $PWD/bench:/b -v /path/to/ct2-model:/ct2 -v /path/to/audio:/a \
    --entrypoint /opt/refenv/bin/python kostryukov/npu-whisper:1.3.4 \
    /b/probe_decoder_ct2.py --ct2 /ct2 /a/clip.wav ...

ctranslate2 is not in the image; put it on PYTHONPATH (pip --target) first.
"""

import argparse
import statistics
import time

import ctranslate2
import numpy as np
import torch
import whisper
from whisper.tokenizer import get_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--ct2", required=True, help="converted CTranslate2 model dir")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="pt")
    ap.add_argument("--compute-types",
                    default="float32,bfloat16,int8_float32,int8_bfloat16,int8")
    ap.add_argument("--threads", default="8", help="comma-separated")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(int(a.threads.split(",")[0]))
    ref = whisper.load_model(a.model, device="cpu")
    tok = get_tokenizer(True, num_languages=ref.num_languages,
                        language=a.language, task="transcribe")
    opts = whisper.DecodingOptions(fp16=False, language=a.language,
                                   without_timestamps=True, temperature=0.0)
    clips = []
    for w in a.wavs:
        audio = whisper.pad_or_trim(whisper.load_audio(w))
        mel = whisper.log_mel_spectrogram(audio, n_mels=ref.dims.n_mels)
        feats = ref.encoder(mel.unsqueeze(0))
        clips.append((w.rsplit("/", 1)[-1], feats,
                      whisper.decode(ref, feats, opts)[0].text.strip()))
    del ref
    prompt = list(tok.sot_sequence_including_notimestamps)
    print(f"ctranslate2 {ctranslate2.__version__}, "
          f"cpu compute types {sorted(ctranslate2.get_supported_compute_types('cpu'))}")

    print("| clip | compute_type | threads | tokens | total ms | ms/token | CPU s | text = torch fp32 |")
    print("|---|---|---|---|---|---|---|---|")
    runs = [(ct, int(th)) for ct in a.compute_types.split(",")
            for th in a.threads.split(",")]
    for ct, th in runs:
        m = ctranslate2.models.Whisper(a.ct2, device="cpu", compute_type=ct,
                                       intra_threads=th)
        for name, feats, ref_text in clips:
            x = ctranslate2.StorageView.from_array(
                np.ascontiguousarray(feats.numpy(), dtype=np.float32))
            times, cpu, ids = [], [], None
            for i in range(a.runs + 1):            # first run is warm-up
                t0, c0 = time.perf_counter(), time.process_time()
                r = m.generate(x, [prompt], beam_size=1, max_length=448)
                dt = (time.perf_counter() - t0) * 1e3
                if i:
                    times.append(dt)
                    cpu.append(time.process_time() - c0)
                ids = r[0].sequences_ids[0]
            text = tok.decode(ids).strip()
            med = statistics.median(times)
            print(f"| {name} | {ct} | {th} | {len(ids)} | {med:.0f} | "
                  f"{med / max(1, len(ids)):.1f} | {statistics.median(cpu):.2f} | "
                  f"{text == ref_text} |", flush=True)
            if text != ref_text:
                print(f"|  |  |  |  |  |  |  | `{text[:120]}` |")
        del m


if __name__ == "__main__":
    main()
