"""Step 6: the NPU encoder with weights from the ggml file instead of the
openai checkpoint -- f16 (the file voxtype downloads) and q8_0 dequantized.

  tiny   golden setup of tests/test_encoder_golden.py: jfk mel and torch fp32
         reference from ref.npz, cosine gate 0.999, transcription via
         tools/decode_with.py
  turbo  the service's code path: cosine against the torch fp32 encoder at
         ctx 1500 (jfk), then ct2 text for every clip at its own ctx step

Every source runs on a runtime wiped of the previous source's weight BOs, so
nothing uploaded for one can be reused by the next.

Run in the service image with stt stopped, mounting the probe dir at /probe,
ggml-tiny.bin / ggml-tiny-q8_0.bin / ggml-large-v3-turbo-q8_0.bin at /models,
voxtype's models dir at /vmodels, the clips at /audio, and src/ggml_weights.py
(not in older images) into /opt/npu-whisper/src; run tools/dump_ref.py first
for ref.npz.
"""
import json, os, subprocess, sys, time, wave
sys.path[:0] = ["/opt/npu-whisper/serve", "/opt/npu-whisper/src"]
import numpy as np
from ggml_weights import load_encoder

MODELS = "/models"
OUT = {}


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def weight_diff(Wref, Wg):
    keys = sorted(k for k in Wref if k in Wg and not k.endswith("_T"))
    missing = sorted(k for k in Wref if not k.endswith("_T") and k not in Wg)
    worst, worst_k, cos_min = 0.0, None, 1.0
    for k in keys:
        a, b = Wref[k].astype(np.float32), Wg[k].reshape(Wref[k].shape)
        r = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-12))
        if r > worst: worst, worst_k = r, k
        cos_min = min(cos_min, cosine(a, b))
    return dict(tensors=len(keys), missing=missing, max_rel=worst, worst=worst_k, min_cos=cos_min)


def add_T(W):
    for k in [k for k in list(W) if k.endswith(".weight") and W[k].ndim == 2
              and not k.endswith("_ln.weight")]:
        W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)
    return W


def wipe(rt):
    rt._chains.clear(); rt._scratch.clear(); rt._scratch_owner.clear()
    while rt._live:
        rt._evict_lru()
    rt._wbo.clear(); rt._known.clear(); rt._keep.clear(); rt._views.clear()


def tiny():
    import npu_whisper_encoder as E
    import rawxrt
    z = np.load("/opt/npu-whisper/ref.npz")
    Wck = {k: z[k].astype(np.float32) for k in z.files}
    mel, ref = Wck.pop("mel"), Wck.pop("ref_out")
    rt = rawxrt.RawRT()
    res = {}
    for src in ("checkpoint", "ggml-f16", "ggml-q8_0"):
        if src == "checkpoint":
            W, info = dict(Wck), {}
        else:
            W, hp, types = load_encoder(f"{MODELS}/ggml-tiny{'' if src == 'ggml-f16' else '-q8_0'}.bin", with_types=True)
            info = weight_diff(Wck, W)
            info["types"] = sorted(set(types.values()))
        add_T(W)
        wipe(rt)
        rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)])
        run = lambda: E.encoder(W, mel, rt.matmul, rt.matmul, prof={}, attn_fn=rt.attn_chain)
        run()
        out = run()
        path = f"/tmp/out_{src}.npy"
        np.save(path, out.astype(np.float32))
        r = subprocess.run(["/opt/npu-whisper/refenv/bin/python", "/opt/npu-whisper/tools/decode_with.py", path],
                           capture_output=True, text=True, cwd="/opt/npu-whisper")
        lines = [l.strip().strip("'") for l in r.stdout.splitlines() if l.strip().startswith("'")]
        res[src] = dict(cos_vs_torch_fp32=cosine(out, ref), gate=cosine(out, ref) >= 0.999,
                        text=lines[-1] if lines else r.stderr[-200:], match_cpu="\nMATCH" in r.stdout,
                        fallbacks=list(rt.fallbacks), **info)
        print("tiny", src, json.dumps(res[src], ensure_ascii=False), flush=True)
    OUT["tiny"] = res


def turbo():
    import npu_stt_server as S
    (S.TORCH, S.WHISPER, S.E, S.RT, S.MODEL_OBJ, S.HOLDER, S.W,
     S.N_LAYER, S.N_HEAD) = S.load_backend()
    Wck = {k: v for k, v in S.W.items() if not k.endswith("_T")}
    clips = {"curto": ("/audio/pt-curto.wav", "pt", 512), "jfk": ("/opt/npu-whisper/tests/data/jfk.wav", "en", 768),
             "longo": ("/audio/pt-longo.wav", "pt", 1280)}
    mels = {}
    for name, (p, lang, c) in clips.items():
        with wave.open(p) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        mels[name] = S.WHISPER.log_mel_spectrogram(S.TORCH.from_numpy(S.WHISPER.pad_or_trim(audio)), n_mels=S.N_MELS)
    # torch fp32 reference for jfk at the full context
    m = S.WHISPER.load_model(S.MODEL).eval()
    with S.TORCH.no_grad():
        ref = m.encoder(mels["jfk"].unsqueeze(0)).squeeze(0).numpy()
    del m
    kw = dict(prof={}, npu_softmax=False, n_layer=S.N_LAYER, n_head=S.N_HEAD,
              attn_fn=S.RT.attn_chain, mlp_fn=S.RT.mlp, fold_bias=True)
    res = {}
    for src in ("checkpoint", "ggml-f16", "ggml-q8_0"):
        if src == "checkpoint":
            W, info = dict(Wck), {}
        else:
            f = "ggml-large-v3-turbo.bin" if src == "ggml-f16" else "ggml-large-v3-turbo-q8_0.bin"
            W, hp, types = load_encoder(("/vmodels/" if src == "ggml-f16" else f"{MODELS}/") + f, with_types=True)
            info = weight_diff(Wck, W)
            info["types"] = sorted(set(types.values()))
        add_T(W)
        extra = S.E.prepare_mlp_chunks(W, S.N_LAYER, W["w.ln_post.weight"].shape[0], 0)
        wipe(S.RT)
        S.RT.register_weights([v for v in W.values() if isinstance(v, np.ndarray)] + extra)
        S.W = W
        S._state["ctx"] = None
        r = dict(info)
        mel1500 = np.ascontiguousarray(mels["jfk"].numpy().astype(np.float32)[:, :3000])
        S.E.encoder(W, mel1500, S.RT.matmul, S.RT.matmul, **kw)
        out = S.E.encoder(W, mel1500, S.RT.matmul, S.RT.matmul, **kw)
        r["cos_vs_torch_fp32_jfk1500"] = cosine(out, ref)
        r["gate"] = r["cos_vs_torch_fp32_jfk1500"] >= 0.999
        S._state["ctx"] = 1500      # so the next ensure_geometry tears this one down
        for name, (p, lang, c) in clips.items():
            mel_np = np.ascontiguousarray(mels[name].numpy().astype(np.float32)[:, :2 * c])
            S.ensure_geometry(c); S._state["ctx"] = c
            feats = S.E.encoder(W, mel_np, S.RT.matmul, S.RT.matmul, **kw)
            r[f"text_{name}_{c}"] = S.CT2.decode(feats, S._tokenizer(lang))
        r["fallbacks"] = list(S.RT.fallbacks)
        res[src] = r
        print("turbo", src, json.dumps(r, ensure_ascii=False), flush=True)
        del W, extra
    OUT["turbo"] = res


for part in sys.argv[1:] or ["tiny", "turbo"]:
    {"tiny": tiny, "turbo": turbo}[part]()
json.dump(OUT, open(f"/probe/golden_ggml_{int(time.time())}.json", "w"), ensure_ascii=False, indent=1)
