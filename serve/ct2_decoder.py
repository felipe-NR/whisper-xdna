"""CTranslate2 decoder for the NPU encoder's output.

The torch decoder reads every weight in fp32 at every token, on a load that is
bound by memory bandwidth, not arithmetic: 22 ms/token for large-v3-turbo on a
Ryzen 7 8700G. CTranslate2 int8 reads a quarter of the bytes and decodes the
same tokens in 5-6 ms (docs/decoder.md). Its generate() takes encoder output
directly, so the NPU encoder stays as it is and only this half changes.

Encoder and decoder must come from ONE checkpoint: a whisper decoder handed
another checkpoint's features does not fail, it confabulates (HISTORY.md). So
there is no downloaded "converted model" here. The CTranslate2 model is built
on first start from the checkpoint the service runs, named by that file's
SHA-256 -- the digest openai-whisper checks the file against whenever it loads
it -- and cached under it. A different checkpoint is a different cache entry;
it can never be served by a stale one.

The conversion fills ctranslate2's WhisperSpec by hand, mirroring what its
Transformers loader does for WhisperForConditionalGeneration, from openai-
whisper module names instead of Hugging Face ones. Decoding options mirror
whisper.decode as serve/ calls it: greedy, no timestamps, the "-1" non-speech
suppression set, blank suppression on the first step, at most 224 new tokens.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np

# Bump when the conversion itself changes, so cached models are rebuilt.
CONVERTER_REVISION = 1
# whisper.decode samples at most n_text_ctx // 2 tokens after the prompt.
SAMPLE_LEN = 224


def _bytes_to_unicode():
    """GPT-2's byte -> printable-character table, as Hugging Face uses it.

    tiktoken ranks are raw bytes; CTranslate2 wants one unique string per id.
    Using the same table as the Hugging Face tokenizer gives the same strings
    as a model converted from there, which makes the two comparable.
    """
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def vocabulary(tokenizer):
    enc = tokenizer.encoding
    special = {i: s for s, i in enc._special_tokens.items()}
    b2u = _bytes_to_unicode()
    return [special[i] if i in special
            else "".join(b2u[b] for b in enc.decode_single_token_bytes(i))
            for i in range(enc.n_vocab)]


def suppress_ids(tokenizer):
    """What whisper.decode suppresses for suppress_tokens="-1"."""
    ids = set(tokenizer.non_speech_tokens)
    ids.update([tokenizer.transcribe, tokenizer.translate, tokenizer.sot,
                tokenizer.sot_prev, tokenizer.sot_lm])
    if tokenizer.no_speech is not None:
        ids.add(tokenizer.no_speech)
    return sorted(ids)


def checkpoint_sha256(name):
    """The digest openai-whisper verifies the download against.

    For a published model name it is part of the download URL; for a local
    checkpoint path it is computed.
    """
    import whisper
    url = whisper._MODELS.get(name)
    if url:
        return url.split("/")[-2]
    h = hashlib.sha256()
    with open(name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_spec(model, tokenizer):
    from ctranslate2.converters import utils
    from ctranslate2.specs import common_spec, whisper_spec

    # Copies, never the parameters themselves. CTranslate2's int8 quantisation
    # does `value = value.to("float32").numpy(); value *= scale`: on an fp32
    # tensor .to() is a no-op and .numpy() shares memory, so it rescales the
    # live torch weights in place (measured: |delta| up to 127 on the token
    # embedding, conv1 and fc1). Its own HF converter gets away with it because
    # that model is thrown away afterwards; this one is the service's model.
    def t(p):
        return p.detach().clone()

    def lin(spec, m):
        spec.weight = t(m.weight)
        if m.bias is not None:
            spec.bias = t(m.bias)

    def ln(spec, m):
        spec.gamma = t(m.weight)
        spec.beta = t(m.bias)

    def attn(spec, a, self_attention):
        # whisper's key projection has no bias; fuse_linear zero-fills it.
        q, k, v = (common_spec.LinearSpec() for _ in range(3))
        lin(q, a.query)
        lin(k, a.key)
        lin(v, a.value)
        if self_attention:
            utils.fuse_linear(spec.linear[0], [q, k, v])
        else:
            utils.fuse_linear(spec.linear[0], [q])
            utils.fuse_linear(spec.linear[1], [k, v])
        lin(spec.linear[-1], a.out)

    def blocks(layer_specs, layers, cross):
        for s, b in zip(layer_specs, layers):
            attn(s.self_attention, b.attn, True)
            ln(s.self_attention.layer_norm, b.attn_ln)
            if cross:
                attn(s.attention, b.cross_attn, False)
                ln(s.attention.layer_norm, b.cross_attn_ln)
            lin(s.ffn.linear_0, b.mlp[0])
            lin(s.ffn.linear_1, b.mlp[2])
            ln(s.ffn.layer_norm, b.mlp_ln)

    d = model.dims
    enc, dec = model.encoder, model.decoder
    spec = whisper_spec.WhisperSpec(d.n_audio_layer, d.n_audio_head,
                                    d.n_text_layer, d.n_text_head)

    for s, c in ((spec.encoder.conv1, enc.conv1), (spec.encoder.conv2, enc.conv2)):
        s.weight, s.bias = t(c.weight), t(c.bias)
    spec.encoder.position_encodings.encodings = t(enc.positional_embedding)
    ln(spec.encoder.layer_norm, enc.ln_post)
    blocks(spec.encoder.layer, enc.blocks, cross=False)

    spec.decoder.embeddings.weight = t(dec.token_embedding.weight)
    spec.decoder.position_encodings.encodings = t(dec.positional_embedding)
    ln(spec.decoder.layer_norm, dec.ln)
    blocks(spec.decoder.layer, dec.blocks, cross=True)
    spec.decoder.projection.weight = t(dec.token_embedding.weight)   # tied

    spec.config.suppress_ids = suppress_ids(tokenizer)
    spec.config.suppress_ids_begin = tokenizer.encode(" ") + [tokenizer.eot]
    spec.config.lang_ids = list(tokenizer.all_language_tokens)
    # Only used for word timestamps, which this service does not produce: the
    # default CTranslate2 applies when a checkpoint names none (last half).
    spec.config.alignment_heads = [(layer, head)
                                   for layer in range(d.n_text_layer // 2, d.n_text_layer)
                                   for head in range(d.n_text_head)]

    vocab = vocabulary(tokenizer)
    if len(vocab) != d.n_vocab:
        raise RuntimeError(f"tokenizer has {len(vocab)} ids, checkpoint {d.n_vocab}")
    spec.register_vocabulary(vocab)
    return spec


def cache_path(name, cache_dir, quantization="int8"):
    sha = checkpoint_sha256(name)
    tag = os.path.basename(str(name)).removesuffix(".pt")
    return os.path.join(cache_dir,
                        f"{tag}-{sha[:16]}-{quantization}-r{CONVERTER_REVISION}"), sha


def convert(model, tokenizer, name, cache_dir, quantization="int8", log=print):
    """Convert `model` (a whisper.Whisper still carrying its torch encoder --
    CTranslate2 has no decoder-only Whisper) into the cache, in this process.

    Writes into a temporary directory and renames it into place, so an
    interrupted run leaves no half model behind. Returns the directory.
    """
    from ctranslate2.converters.converter import Converter

    out, sha = cache_path(name, cache_dir, quantization)

    class _FromOpenAI(Converter):
        def _load(self):
            return _build_spec(model, tokenizer)

    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{out}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.perf_counter()
    _FromOpenAI().convert(tmp, quantization=quantization, force=True)
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump({"checkpoint": str(name), "checkpoint_sha256": sha,
                   "quantization": quantization,
                   "converter_revision": CONVERTER_REVISION}, f, indent=2)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    log(f"ct2 decoder: converted {name} ({sha[:16]}) to {quantization} "
        f"in {time.perf_counter() - t0:.1f} s -> {out}")
    return out


def ensure_model(name, cache_dir, quantization="int8", log=print):
    """Return the cached CTranslate2 model for checkpoint `name`, converting
    it first if needed -- in a child process.

    Not in this one, because the conversion holds an fp32 copy of every weight
    (~3 GB for large-v3-turbo) and the allocator keeps those pages after they
    are freed: measured, a service that converted in-process sat at 11.0 GiB
    against 8.0 GiB for one that loaded the cached model, and malloc_trim did
    not give it back. A child returns all of it when it exits.

    The child loads `name` through whisper.load_model, which checks the file
    against the same SHA-256 the cache entry is named after, so it converts the
    checkpoint this service runs and no other.
    """
    import subprocess
    import sys

    out, _ = cache_path(name, cache_dir, quantization)
    if os.path.exists(os.path.join(out, "manifest.json")):
        log(f"ct2 decoder: cached {out}")
        return out
    log(f"ct2 decoder: converting {name} to {quantization} (first start only)")
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                    "--model", str(name), "--cache-dir", cache_dir,
                    "--quantization", quantization], check=True)
    if not os.path.exists(os.path.join(out, "manifest.json")):
        raise RuntimeError(f"ct2 conversion finished but {out} is missing")
    return out


def _main():
    import argparse

    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer

    ap = argparse.ArgumentParser(description="convert a whisper checkpoint for ct2")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--quantization", default="int8")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model = whisper.load_model(a.model, device="cpu")
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                        task="transcribe")
    convert(model, tok, a.model, a.cache_dir, a.quantization,
            log=lambda m: print(m, flush=True))


class CT2Decoder:
    """Greedy decode of encoder output, matching whisper.decode in serve/."""

    def __init__(self, model_dir, compute_type="int8", threads=8):
        import ctranslate2
        self._sv = ctranslate2.StorageView
        self.model = ctranslate2.models.Whisper(
            model_dir, device="cpu", compute_type=compute_type,
            intra_threads=threads)
        self.compute_type = compute_type
        self.threads = threads

    def decode(self, feats, tokenizer):
        """feats: (n_ctx, d_model) float32 from the encoder -> text."""
        prompt = list(tokenizer.sot_sequence_including_notimestamps)
        x = self._sv.from_array(np.ascontiguousarray(feats[None], dtype=np.float32))
        # Measured: Whisper.generate returns max_length // 2 new tokens (112 for
        # 224, 224 for 448) -- it applies whisper's own n_text_ctx // 2 rule
        # itself. 2 * SAMPLE_LEN is what stops a looping decode exactly where
        # whisper.decode stops it.
        r = self.model.generate(x, [prompt], beam_size=1, suppress_blank=True,
                                suppress_tokens=[-1], max_length=2 * SAMPLE_LEN)
        return tokenizer.decode(r[0].sequences_ids[0]).strip()

    def language_probs(self, feats):
        """{language code: probability} for one window of encoder output.

        CTranslate2's detect_language runs the same single decoder step over
        the language tokens that whisper.detect_language does.
        """
        x = self._sv.from_array(np.ascontiguousarray(feats[None], dtype=np.float32))
        return {tok[2:-2]: p for tok, p in self.model.detect_language(x)[0]}


if __name__ == "__main__":
    _main()
