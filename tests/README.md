# Tests

Every automated test of the tree, and how to run them. `tests/run_all.sh` runs
all of them in dependency order through `compose.yml` + `compose.ct2.yml`, and
puts the `stt` service back as `.env` describes it when it exits, pass or fail.

```bash
tests/run_all.sh                              # host build cpu npu serve
tests/run_all.sh host cpu                     # some stages only
CLIPS=~/audio tests/run_all.sh serve          # + your own .wav/.txt pairs
```

| stage | test | runs in | needs | what it guards |
|---|---|---|---|---|
| host | `test_compare_dictation.py` | host, stdlib | nothing | `tools/compare_dictation.py`: WER, normalisation, `audio_ctx` sizing, voxtype log parsing (text from stdout, never the truncated log line) |
| build | — | `docker compose build stt` | the base image | the ct2 overlay image builds (`serve/ct2.Dockerfile`) |
| cpu | `test_ct2_unit.py` | `test-cpu` (ct2 image) | CPU, whisper-tiny | `serve/ct2_decoder.py`: suppression set equal to `whisper.decode`'s own, vocabulary, conversion leaves the model untouched, cache key/hit/miss in a child, `max_length` semantics, float32 text identical to `whisper.decode`, the `NPU_STT_DECODER` knob |
| cpu | `test_ct2_decoder.py` | `test-cpu` | CPU, large-v3-turbo | int8 text identical to `whisper.decode` on `jfk.wav` at ctx 1500 and 512 (pass more clips and `--ctx` by hand) |
| npu | `test_rowops.py` | `test-npu` (ironenv) | the NPU, `stt` stopped | row-wise softmax/layernorm kernels |
| npu | `test_chain_acc.py` | `test-npu` | the NPU | accuracy cost of the QK→softmax→AV chain |
| npu | `test_encoder_golden.py` | `test-npu` | the NPU, `ref.npz` (made by `tools/dump_ref.py` in the same stage) | the acceptance gate: cosine, bit-identical golden, chain vs host, fallback, the 6-context ceiling, `jfk.wav` transcription |
| serve | `test_serve_http.py` | host, stdlib | `stt` up | `/health` names the decoder; `/inference` and `/v1/audio/transcriptions` transcribe `jfk.wav` on the NPU with timing headers; short `audio_ctx`; 404/400 paths leave it healthy; optional clips within `--max-wer`; `--save`/`--against` for A/B |

The serve stage runs `test_serve_http.py` twice: with `NPU_STT_DECODER=torch`
(saving every transcript), then with `ct2`, which must reproduce each one
exactly. The two cannot run side by side: the NPU has 6 `hw_context`s and one
service holds most of them. For the same reason the npu stage stops `stt`.

## Results on the reference host

Ryzen 7 8700G (XDNA1, firmware 1.5.5.391), kernel 7.2.5, image
`npu-whisper:ct2` on `kostryukov/npu-whisper:1.3.4`, 2026-10-17:

| stage | result |
|---|---|
| host | 17/17 |
| cpu | 19/19 (`test_ct2_unit.py`), OK (`test_ct2_decoder.py`) |
| npu | rowops PASS, chain_acc PASS, encoder golden 14/14 (bit-identical) |
| serve-torch | 15/15 (with the two pt-BR clips of `docs/research.md`) |
| serve-ct2 | 20/20, all 5 transcripts identical to the torch run |

The same stages (`cpu npu serve`) also pass, with the same counts, on the image
built from the main `Dockerfile` (`WITH_REFERENCE=1 WITH_CT2=1`, commit
`ea33950`, 2026-10-17), which carries XRT 2.25.0 and a newer `ironenv`. The
encoder golden is still bit-identical there. See "Pinned versions" in the
top-level README.

## Two things the tests caught

- **int8 on whisper-tiny flips a comma.** "for you, ask" against "for you ask"
  at ctx=1500: a near-tie, not a conversion error. Converted to float32 the
  same model gives identical text, so `test_ct2_unit.py` checks the conversion
  in float32 (exact) and int8 for the same words. Exact int8 parity is
  required on large-v3-turbo, where it holds on every clip tried.
- **A short context has to be encoded short.** Cutting a 1500-position
  encoding to 512 positions makes both decoders loop on garbage. What the NPU
  (and whisper.cpp's `-ac`) actually does is encode the first `2·ctx` mel frames
  against `ctx` positions, and that is what the decoder tests feed.
