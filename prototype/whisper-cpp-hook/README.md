# Prototype: whisper.cpp's decoder fed by the NPU encoder

The question for a native voxtype integration: can whisper.cpp, the decoder
voxtype already ships, take the encoder output this repo computes on the XDNA1
NPU and produce the same text at a usable speed? Yes.

## The hook

`0001-encoder-callback-v1.9.4.patch` adds a runtime callback to whisper.cpp
(about 40 lines in `src/whisper.cpp` and `include/whisper.h`):

```c
typedef bool (*whisper_encoder_callback)(
        const float * mel, int n_mel, int n_ctx,
              float * out, int n_state, void * user_data);

void whisper_set_encoder_callback(struct whisper_context * ctx,
                                  whisper_encoder_callback cb, void * user_data);
void whisper_set_encoder_callback_with_state(struct whisper_context * ctx,
                                             struct whisper_state * state,
                                             whisper_encoder_callback cb, void * user_data);
```

- **What the callback does.** With a callback set, `whisper_encode_external()`
  reports true, and whisper.cpp skips the ggml conv and encoder graphs. For
  each window, the callback receives whisper.cpp's own log-mel (`n_mel` rows of
  `2*n_ctx` frames) and writes the encoder output after `ln_post` (`n_ctx`
  rows of `n_state` floats).
- **What whisper.cpp still does.** It computes the cross-attention K/V, and
  then decoding, language detection, timestamps and the rest run unchanged.
- **Where it comes from.** It is the mechanism the CoreML, OpenVINO and
  VitisAI encoders already use, exposed as a callback instead of a compile-time
  backend. Like OpenVINO, it can be attached to a state after it is created.
- **The 1.8.3 variant.** `0001-encoder-callback-v1.8.3.patch` is the same
  change against whisper.cpp 1.8.3, the version bundled by whisper-rs-sys 0.15
  and used by voxtype 1.1.1. Only one line differs, because 1.8.3 has no
  VitisAI term, and libwhisper builds with it.

The patch also adds `examples/npu-feed/` (`whisper-npu-feed`). The program
decodes a WAV whose encoder output comes from a `.npy` file, or runs the ggml
encoder for comparison, and saves the mel whisper.cpp computed.

## Results

Setup:
- Ryzen 7 8700G, `large-v3-turbo`, 2026-10-18, language fixed.
- NPU encoder output from `dump_encoder_output.py` (whisper-xdna 9a1ba08,
  service code).
- whisper.cpp v1.9.4 + this patch, CPU, `GGML_NATIVE=ON`, built in
  `debian:13`. Greedy, no timestamps, no temperature fallback.
- Times are the median of 5 `whisper_full` runs after a warm-up. "cross" is
  whisper.cpp's cross-attention K/V graph.

**Text.** In all six cases (6 s @512, 11 s @768, 20 s @1280, and each clip at
1500), whisper.cpp's decoder on the NPU features returns exactly the text of
the service's ct2 decoder on the same features. That is with the f16 model.
The q8_0 model gives the same text in the three cases at their own ctx step,
and so does whisper.cpp with its own CPU encoder (both checked at 512, 768
and 1280).

**Mel.** whisper.cpp's log-mel and the torch mel the service feeds the NPU
differ by at most 4.5e-5 (values within −0.75…1.28), far below bf16 resolution
(~4e-3). A native integration can hand whisper.cpp's mel straight to the NPU.

**Everything after the encoder** (whisper.cpp: mel + cross K/V + decode;
service: ct2 decode):

| phrase (ctx) | f16, 8 threads | q8_0, 8 threads | q8_0, 4 threads | ct2 int8, 8 threads |
|---|---|---|---|---|
| 6 s pt (512) | 225 ms (cross 40) | **138 ms** (cross 26) | 166 ms | 126 ms |
| 11 s en (768) | 240 ms (cross 61) | **151 ms** (cross 39) | 190 ms | 126 ms |
| 20 s pt (1280) | 538 ms (cross 102) | **339 ms** (cross 67) | 408 ms | 323 ms |

With a q8_0 model, whisper.cpp's decoder runs within 10-25 ms of ct2. The
ggml file voxtype downloads is f16, and 874 MB as q8_0.

**Encoder, for scale:** whisper.cpp's own CPU encoder at the same ctx takes
2.09 s (512), 3.20 s (768) and 5.55 s (1280). The NPU takes 1.18, 1.99 and
3.08 s.

## Caveats for a real integration

- **Language detection.** whisper.cpp runs the encoder a second time when it
  auto-detects the language (measured from voxtype: +1.08 s on the iGPU). With
  this hook, the callback would fire twice and run the NPU encoder twice. The
  integration has to cache by mel window, or detect on the same pass as the
  service does.
- **Weights.** In this prototype the NPU features come from the service, which
  loads the encoder weights from the openai-whisper checkpoint. Loading them
  from the ggml f16 file instead is INBOX step 6.
- **Encoder runtime.** The callback here copies a file. The real one drives
  the NPU through XRT. That runtime is the Python code in `src/`, and it has
  to be ported to C++.
