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

## Encoder weights from the ggml file

`src/ggml_weights.py` reads the encoder tensors straight out of a whisper.cpp
ggml model file (f32, f16 and q8_0, numpy only), keyed the way the rest of
the repo expects. `golden_ggml.py` runs each weight source through the NPU
encoder on a runtime wiped of the previous source's weight BOs.

- **ggml f16 is the checkpoint, bit for bit.** Both openai checkpoints are
  stored in fp16, and so is the ggml file voxtype downloads. For whisper-tiny
  (67 encoder tensors) and large-v3-turbo (487) the largest difference is 0.
  The NPU output and the text are the same as with the checkpoint's weights.
  A native integration can take its encoder weights from the one model file
  whisper.cpp already loads, at no accuracy cost.
- **q8_0, dequantized to bf16 at upload, costs a little and changes no text.**
  - Weight cosine is ≥ 0.99996 for every tensor.
  - whisper-tiny: cosine against torch fp32 drops from 0.99969 to 0.99940,
    still above the 0.999 gate, and the transcription matches the CPU
    reference.
  - large-v3-turbo at ctx 1500: 0.99585 → 0.99516. The 0.999 gate does not
    apply at this depth (see the main README), so text is the criterion. The
    6 s @512, 11 s @768 and 20 s @1280 transcripts are identical to those
    from the checkpoint's weights.
  - So a single 874 MB q8_0 file could feed both the NPU encoder and the
    decoder (see the table above). That is checked on four clips, not more.

## Caveats for a real integration

- **Reduced `audio_ctx` hallucinates on edge cases.** This is whisper.cpp's
  own behaviour, not the NPU's: Vulkan with `context_window_optimization`
  (432 / 1232) produces the same errors. Reported by the voxtype session,
  which ran the native NPU encoder at the ladder ctx:
  - jfk cut to 6 s, mid-sentence, at 512 ends "…ask not what your country is.";
  - jfk twice back to back (22 s) at 1280 repeats the sentence three times.

  Full ctx (1500) gets both right. On real clips the ladder ctx and 1500 gave
  identical text here (6 s @512, 11 s @768, 20 s @1280), so there is no
  evidence yet that the selection margin is too small for real dictation. A
  safety net would be to retry at full ctx when the output repeats a long
  n-gram.
- **No stale output in rawxrt.** The voxtype loader saw the first 64 bytes of
  an output BO keep the previous launch's contents until it flushed that cache
  line before each launch. rawxrt does not reproduce this. Every device path
  was on, NPU GELU included: whisper-tiny @512 and large-v3-turbo @768 on jfk
  gave bit-identical output in 6 fresh processes each, and pass 1 equalled
  pass 2 in every process. The likely difference: rawxrt never writes into an
  output BO from the host, so no dirty CPU line can be written back over the
  NPU's result. Any output BO the host has written to (zero-fill, reuse as an
  input) needs a `sync(TO_DEVICE)` before the launch.
- **Language detection.** whisper.cpp runs the encoder a second time when it
  auto-detects the language (measured from voxtype: +1.08 s on the iGPU). With
  this hook, the callback would fire twice and run the NPU encoder twice. The
  integration has to cache by mel window, or detect on the same pass as the
  service does.
- **Weights.** In this prototype the NPU features come from the service, which
  loads its weights from the openai checkpoint. That is identical to the ggml
  f16 file (see above). The service itself still reads the checkpoint.
- **Encoder runtime.** The callback here copies a file. The real one drives
  the NPU through XRT, and that runtime is the Python code in `src/` today. The
  design agreed for voxtype splits it in two:
  - a thin C shim over XRT (`libvoxtype_xrt.so`: about 15 functions over
    opaque handles, plus one `vxrt_npu_run`), loaded with dlopen;
  - the encoder graph in Rust inside voxtype: chaining, runlists, the ctx
    ladder, the host ops, and weight BOs built from the ggml tensors.

  A self-contained C++ encoder library behind the same callback is the
  alternative. It would be reusable outside voxtype, but would move most of
  the code out of what voxtype's maintainer reviews.
