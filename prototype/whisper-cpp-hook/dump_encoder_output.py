"""Dump the NPU encoder output (and the torch mel it was fed) per clip and ctx.

Runs the service's own code in the service image while stt is stopped (the
NPU's hw_contexts are device-wide). The mel is built exactly as
serve/npu_stt_server.py::_window builds it, then E.encoder runs on it.
Writes /probe/feats/<clip>_<ctx>.npy (float32, ctx x n_state) and
<clip>_<ctx>_mel.npy (n_mels x 2*ctx), plus a ct2 decode of the same
features as the text/latency baseline (index.json).

  docker compose stop stt
  docker compose run --rm --no-deps -v <out>:/probe -v <clips>:/audio:ro stt \
    bash -c 'export PYTHONPATH=/usr/lib/python3/dist-packages \\
               PEANO_INSTALL_DIR=/opt/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie
             cd /opt/npu-whisper && refenv/bin/python -W ignore /probe/dump_encoder_output.py'
  docker compose start stt

<clips> holds pt-curto.wav and pt-longo.wav (not versioned); jfk.wav comes
from tests/data in the image.
"""
import json, os, sys, time, wave
sys.path[:0] = ["/opt/npu-whisper/serve", "/opt/npu-whisper/src"]
import numpy as np
import npu_stt_server as S

(S.TORCH, S.WHISPER, S.E, S.RT, S.MODEL_OBJ, S.HOLDER, S.W,
 S.N_LAYER, S.N_HEAD) = S.load_backend()
os.makedirs("/probe/feats", exist_ok=True)

CLIPS = {"curto": ("/audio/pt-curto.wav", "pt"), "jfk": ("/opt/npu-whisper/tests/data/jfk.wav", "en"),
         "longo": ("/audio/pt-longo.wav", "pt")}
PLAN = [("curto", 512), ("jfk", 768), ("longo", 1280), ("curto", 1500), ("jfk", 1500), ("longo", 1500)]
out = []
for clip, ctx in PLAN:
    path, lang = CLIPS[clip]
    with wave.open(path) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    S.ensure_geometry(ctx)
    padded = S.WHISPER.pad_or_trim(audio)
    mel = S.WHISPER.log_mel_spectrogram(S.TORCH.from_numpy(padded), n_mels=S.N_MELS)
    mel_np = np.ascontiguousarray(mel.numpy().astype(np.float32)[:, :2 * ctx])
    kw = dict(prof={}, npu_softmax=False, n_layer=S.N_LAYER, n_head=S.N_HEAD,
              attn_fn=S.RT.attn_chain, mlp_fn=S.RT.mlp, fold_bias=True)
    S.E.encoder(S.W, mel_np, S.RT.matmul, S.RT.matmul, **kw)      # warm this geometry
    S._state["ctx"] = ctx
    t0 = time.perf_counter()
    feats = S.E.encoder(S.W, mel_np, S.RT.matmul, S.RT.matmul, **kw)
    enc_ms = (time.perf_counter() - t0) * 1e3
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    np.save(f"/probe/feats/{clip}_{ctx}.npy", feats)
    np.save(f"/probe/feats/{clip}_{ctx}_mel.npy", mel_np)
    decs = []
    for _ in range(4):
        t0 = time.perf_counter()
        text = S.CT2.decode(feats, S._tokenizer(lang))
        decs.append((time.perf_counter() - t0) * 1e3)
    r = dict(clip=clip, ctx=ctx, lang=lang, shape=list(feats.shape), enc_ms=round(enc_ms),
             ct2_dec_ms=round(sorted(decs[1:])[1]), ct2_text=text, fallbacks=len(S.RT.fallbacks))
    out.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(out, open("/probe/feats/index.json", "w"), ensure_ascii=False, indent=1)
