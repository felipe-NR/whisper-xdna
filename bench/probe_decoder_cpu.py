"""Where the CPU decoder's time goes, and what cheaper arithmetic buys.

The service decodes with `whisper.decode` in fp32 on 16 threads. This probe
holds the encoder output fixed (torch reference encoder, computed once) and
times only the decoder, so every variant answers one question:

  decode/fp32   whisper.decode as serve/ calls it: the baseline
  greedy/fp32   a bare greedy loop with the same KV-cache hooks, to separate
                DecodingTask overhead from arithmetic
  greedy/bf16   decoder weights and activations in bf16 (oneDNN, avx512_bf16)
  greedy/int8   dynamic int8 Linear layers (fbgemm, VNNI); the tied logits
                projection stays fp32, as nn.Linear swapping cannot reach it

Each runs on every --threads value. Reported per clip: tokens generated, the
first step (prompt + cross-attention K/V over the whole context), the median
per-token step, the total, the CPU time it burned across all threads
(process_time), and whether the text matches the baseline.

  docker run --rm -v whisper-xdna_models:/root/.cache/whisper \
    -v $PWD/bench:/b -v /path/to/audio:/a --entrypoint /opt/refenv/bin/python \
    kostryukov/npu-whisper:1.3.4 /b/probe_decoder_cpu.py /a/clip.wav ...
"""

import argparse
import copy
import statistics
import time

import torch
import whisper
from whisper.tokenizer import get_tokenizer


def greedy(model, tok, feats, max_len=224):
    """Plain greedy decode; returns (tokens, first_ms, per_token_ms list)."""
    cache, hooks = model.install_kv_cache_hooks()
    try:
        prompt = torch.tensor([list(tok.sot_sequence_including_notimestamps)])
        t0 = time.perf_counter()
        logits = model.decoder(prompt, feats, kv_cache=cache)
        first = (time.perf_counter() - t0) * 1e3
        out, steps = [], []
        for _ in range(max_len):
            nxt = int(logits[0, -1].float().argmax())
            if nxt == tok.eot:
                break
            out.append(nxt)
            t0 = time.perf_counter()
            logits = model.decoder(torch.tensor([[nxt]]), feats, kv_cache=cache)
            steps.append((time.perf_counter() - t0) * 1e3)
        return out, first, steps
    finally:
        for h in hooks:
            h.remove()


def as_int8(model):
    """Dynamic int8 on every decoder Linear.

    whisper.model.Linear subclasses nn.Linear only to cast weights to the input
    dtype, which quantize_dynamic's exact-type mapping does not recognise; the
    parameters are identical, so relabelling the class is enough.
    """
    m = copy.deepcopy(model)
    for mod in m.decoder.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.__class__ = torch.nn.Linear
    m.decoder = torch.ao.quantization.quantize_dynamic(
        m.decoder, {torch.nn.Linear}, dtype=torch.qint8)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="pt")
    ap.add_argument("--threads", default="16,8")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    model = whisper.load_model(a.model, device="cpu")
    tok = get_tokenizer(True, num_languages=model.num_languages,
                        language=a.language, task="transcribe")
    feats = {}
    for w in a.wavs:
        audio = whisper.pad_or_trim(whisper.load_audio(w))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        feats[w] = model.encoder(mel.unsqueeze(0))
    del model.encoder                     # only the decoder is under test
    print(f"torch {torch.__version__}, cpu capability "
          f"{torch.backends.cpu.get_cpu_capability()}, "
          f"decoder params {sum(p.numel() for p in model.decoder.parameters())/1e6:.0f} M")

    bf16 = copy.deepcopy(model)
    bf16.decoder.to(torch.bfloat16)
    # whisper's LayerNorm runs in fp32 (x.float()) and needs fp32 parameters.
    for mod in bf16.decoder.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            mod.float()
    int8 = as_int8(model)
    opts = whisper.DecodingOptions(fp16=False, language=a.language,
                                   without_timestamps=True, temperature=0.0)

    print("| clip | variant | threads | tokens | 1st step ms | ms/token | total ms | CPU s | text = baseline |")
    print("|---|---|---|---|---|---|---|---|---|")
    for w, f in feats.items():
        ref_text = whisper.decode(model, f, opts)[0].text   # warm-up + reference
        for th in (int(x) for x in a.threads.split(",")):
            torch.set_num_threads(th)
            # decode/fp32: total only; DecodingTask exposes no per-step timing
            tot, cpu = [], []
            for _ in range(a.runs + 1):
                t0, c0 = time.perf_counter(), time.process_time()
                text = whisper.decode(model, f, opts)[0].text
                tot.append((time.perf_counter() - t0) * 1e3)
                cpu.append(time.process_time() - c0)
            n = len(tok.encode(" " + text.strip()))
            print(f"| {w.rsplit('/', 1)[-1]} | decode/fp32 | {th} | ~{n} | – | – | "
                  f"{statistics.median(tot[1:]):.0f} | {statistics.median(cpu[1:]):.2f} | "
                  f"{text == ref_text} |")
            for name, m, dt in (("greedy/fp32", model, torch.float32),
                                ("greedy/bf16", bf16, torch.bfloat16),
                                ("greedy/int8", int8, torch.float32)):
                x = f.to(dt)
                res, cpu = [], []
                for _ in range(a.runs + 1):
                    c0 = time.process_time()
                    res.append(greedy(m, tok, x))
                    cpu.append(time.process_time() - c0)
                res, cpu = res[1:], cpu[1:]
                out = res[-1][0]
                text = tok.decode(out)
                firsts = [r[1] for r in res]
                steps = [s for r in res for s in r[2]]
                totals = [r[1] + sum(r[2]) for r in res]
                print(f"| {w.rsplit('/', 1)[-1]} | {name} | {th} | {len(out)} | "
                      f"{statistics.median(firsts):.0f} | "
                      f"{statistics.median(steps):.1f} | "
                      f"{statistics.median(totals):.0f} | "
                      f"{statistics.median(cpu):.2f} | "
                      f"{text.strip() == ref_text.strip()} |", flush=True)
                if text.strip() != ref_text.strip():
                    print(f"|  |  |  |  |  |  |  |  | `{text.strip()[:120]}` |")


if __name__ == "__main__":
    main()
