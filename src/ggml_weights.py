"""Encoder weights straight from a whisper.cpp ggml model file.

A native integration has one model file -- the ggml one whisper.cpp already
loads for the decoder -- so the NPU encoder has to take its weights from there
rather than from the openai-whisper checkpoint. This reads the legacy ggml
container whisper.cpp writes (convert-pt-to-ggml.py, quantize) and returns
the encoder tensors as fp32 numpy arrays keyed like the checkpoint's
model.encoder.state_dict() with the "w." prefix the rest of this repo uses.

Supported tensor types: f32, f16 and q8_0 (dequantized here; the NPU path
converts to bf16 at upload either way). Numpy only.
"""
import struct

import numpy as np

GGML_MAGIC = 0x67676D6C
HPARAMS = ("n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head",
           "n_audio_layer", "n_text_ctx", "n_text_state", "n_text_head",
           "n_text_layer", "n_mels", "ftype")
F32, F16, Q8_0 = 0, 1, 8
QK8_0 = 32


def _dequant_q8_0(raw, n):
    # block: fp16 scale d, then 32 int8 quants; value = d * q
    blocks = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("q", "i1", QK8_0)]))
    return (blocks["d"].astype(np.float32)[:, None] * blocks["q"].astype(np.float32)).reshape(n)


def load_encoder(path, with_types=False):
    """-> (W, hparams[, types]): W maps "w.<state_dict key>" -> fp32 ndarray."""
    W, types = {}, {}
    with open(path, "rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
        if magic != GGML_MAGIC:
            raise ValueError(f"{path}: not a ggml whisper model (magic {magic:#x})")
        hp = dict(zip(HPARAMS, struct.unpack("<11i", f.read(44))))
        n_mel, n_fft = struct.unpack("<2i", f.read(8))
        f.seek(n_mel * n_fft * 4, 1)                       # mel filters
        (n_vocab,) = struct.unpack("<i", f.read(4))
        for _ in range(n_vocab):                            # vocab
            (ln,) = struct.unpack("<I", f.read(4))
            f.seek(ln, 1)
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            n_dims, name_len, ttype = struct.unpack("<3i", head)
            ne = struct.unpack(f"<{n_dims}i", f.read(4 * n_dims))
            name = f.read(name_len).decode()
            n = int(np.prod(ne))
            if ttype == F32:
                nbytes = 4 * n
            elif ttype == F16:
                nbytes = 2 * n
            elif ttype == Q8_0:
                nbytes = n // QK8_0 * (2 + QK8_0)
            else:
                raise ValueError(f"{name}: ggml type {ttype} not supported here")
            if not name.startswith("encoder."):
                f.seek(nbytes, 1)
                continue
            raw = f.read(nbytes)
            if ttype == F32:
                a = np.frombuffer(raw, "<f4").astype(np.float32)
            elif ttype == F16:
                a = np.frombuffer(raw, "<f2").astype(np.float32)
            else:
                a = _dequant_q8_0(raw, n)
            # ggml lists dims fastest-first; numpy wants them slowest-first
            a = a.reshape(tuple(reversed(ne)))
            key = name[len("encoder."):]
            if key in ("conv1.bias", "conv2.bias"):
                a = a.reshape(-1)                           # stored as (n_state, 1)
            W["w." + key] = np.ascontiguousarray(a)
            types[key] = {F32: "f32", F16: "f16", Q8_0: "q8_0"}[ttype]
    return (W, hp, types) if with_types else (W, hp)
