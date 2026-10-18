# Kernel pack format (`xdna-whisper-pack/1`)

A kernel pack holds the compiled NPU overlays one Whisper model needs on one
NPU generation, plus a manifest. A native runtime (voxtype's Rust encoder over
a thin XRT shim) reads the manifest instead of this repository's Python.
Weights are **not** in the pack: they come from the whisper.cpp ggml model
file at run time (`src/ggml_weights.py` is the reference reader). A pack
therefore fits any model of the same architecture, fine-tunes included.

- Produced by `tools/export_pack.py`. It runs in the service image and
  verifies on the NPU that one real encoder pass per ladder step opens exactly
  the designs the manifest lists, using weights read from the ggml file.
- Checked by `tools/check_pack.py`, standard library only. A loader must refuse
  everything that tool refuses.
- Example manifests, and golden fixtures for whisper-tiny, are in
  `tests/data/packs/`.

The rule for what goes in JSON: **if changing a number would need a code change
in the loader rather than a new pack, the number belongs in the manifest.**
The prose below explains what the numbers mean.

## Layout

```
<pack_id>/
  manifest.json
  designs/<design>.xclbin    the overlay
  designs/<design>.insts     its instruction stream (uint32 words)
```

The `pack_id` is `<model>-<generation>-<first 12 hex of sha256>`. The hash is
taken over the concatenated `xclbin` and `insts` sha256 strings of every
design, in sorted design-name order. Two packs with the same id are the same
pack.

## Manifest

| field | meaning |
|---|---|
| `schema` | `xdna-whisper-pack/<major>`. A loader rejects a major it does not know. |
| `model` | Encoder hparams: `n_mels`, `n_audio_ctx`, `n_audio_state`, `n_audio_head`, `n_audio_layer`. They must equal the ggml file's header, otherwise refuse. `weight_types` lists the encoder tensor types the pack accepts (`f16`, `q8_0`; `f32` is always fine, since norms and biases are f32). Any other type is an error. `validated_with` names the file and sha256 the pack was verified with; a different file is only worth a note. |
| `device` | `generation`, PCI ids, `vbnv`, `max_hw_contexts` (6 on XDNA1), `firmware_tested`, `xrt_min` (the lowest XRT tested, not a proven minimum), `xrt_tested`, and the driver (in-tree, so a kernel release and srcversion rather than a version). |
| `toolchain` | For humans: MLIR-AIE, peano, the XRT the overlays were built against, the whisper-xdna commit. |
| `memory` | `heap_bytes` (the 64 MiB first mapping XRT makes), `max_bo_bytes` (the largest single BO any role allocates), and `memlock_min_bytes`. See "Memory". |
| `abi` | The kernel calling convention. See "Kernel ABI". |
| `encoder_output` | The contract with whisper.cpp: `ctx` rows (padding stripped) × `n_audio_state` columns, f32, after `ln_post`. |
| `selection` | How to choose a ladder step for a clip. See "Ladder selection". |
| `ladder[]` | One entry per context step: `ctx`, `max_audio_secs`, `attn` (shared attention padding), `mlp` (which MLP path), and `roles`, each of which names a design and spells out its operands. |
| `designs{}` | Per design: `kind`, `variant` (rowops), shape fields, dtypes, `kernel` name, and for `xclbin` and `insts` a `path`, `bytes` and `sha256`. |
| `verified` | Present when the exporter ran the on-device check. |

### Integrity

The pack comes from a download, so a loader checks the following **at load
time**, not only after downloading:

- every `path` is exactly `designs/<file>`: not absolute, no `..`, not a symlink;
- every file's size equals `bytes` and its hash equals `sha256`;
- `pack_id` equals the hash rule above;
- every role names a design that exists, of the kind the role requires;
- every design is used by at least one step;
- `memlock_min_bytes` and every `max_audio_secs` follow their rules.

## Kernel ABI

Every kernel is called with opcode 3 and its instruction stream:

```
kernel(3, insts_bo, n_insts, <buffers...>)
  gemm:  insts at argument 1; a, b, c at arguments 3, 4, 5
  rowop: insts at argument 1; in, out at arguments 3, 4
  insts_bo cacheable; operand BOs host_only
```

`abi.kernel_args` holds those **argument indices**. They are not memory
groups. A BO's group is `kernel.group_id(arg_index)` on the kernel it feeds,
queried at run time, and the value is **not a constant**: on this stack it
reads as `(n << 16) | bank`, and `n` changes with every hw_context opened.
Across one export, the GEMM operands read 65536, 196608, 262144 and so on,
and the rowops 131072, 327680 and so on. Only the low half is stable: bank 0
for operands, bank 1 for instruction streams.
`abi.memory_groups_observed` records just those low halves, plus how many
distinct values were seen. Never cache a group across contexts. (Packs
exported before this note carried `group_ids` fields that held the argument
indices under a misleading name. Ignore them.)

That said, the groups matter less than they look: `src/rawxrt.py` allocates
the attention stacks against the softmax kernel and feeds them to the QK and
AV kernels, and allocates weights against one design and feeds them to every
ladder step's design, all with correct results.

A `host_only` BO is not tied to the hw_context it was allocated against. One
BO can feed kernels in other contexts and xclbins, and it survives their
teardown. That is what lets weights stay resident across ladder steps
(`src/rawxrt.py`, `_wbo`).

## Kinds and variants

A closed set per schema version. A loader rejects anything else instead of
guessing.

| kind | variant | semantics | reference |
|---|---|---|---|
| `gemm` | — | `c = a @ b`. bf16 in, f32 out (`mm_*`) or bf16 out (`mmbf_*`, chainable into a rowop). Operands are padded row-major; see each role's `a`/`b`/`c`. | `rawxrt.compile_shape`, `padded_shape` |
| `softmax` | `s3` | Row softmax over `rows × cols` bf16, all columns. | `rowops.softmax_row3` |
| `softmax` | `s4w` | Row softmax over only the first `valid_cols` of each `cols`-wide row. The rest are GEMM padding, which is 0 and not −inf, so including it would leak 1.4% of the probability mass on average and 8.9% at worst. | `rowops.softmax_row4_win` |
| `gelu` | `g4r` | Erf-GELU, bf16 → bf16, round-to-nearest-even, over a fixed `rows × cols` tile. | `rowops.gelu_row4_rne` |
| `gelu` | `g4f` | The same GELU, reading f32 and writing bf16 into a band `out_cols` wide (fc1 → fc2's A). | `rowops.gelu_row4f` |

## Roles

Each ladder step maps roles to designs, in the order one encoder pass runs
them (`src/npu_whisper_encoder.py::encoder`). Dimensions use the model's
`d = n_audio_state`, `nh = n_audio_head`, `hd = d / nh` and `d_ff = 4·d`.

- **Padding.** For every GEMM, `pad` gives the multiples M, K and N are padded
  to (TM·8, TK, TN·4). `Mp`, `Kp` and `Np` are the results. Padding is zero.
  Results are read back unpadded: the first M rows and N columns of `c`.
- **`conv1`, `conv2`.** The convolutions run as im2col GEMMs. For `conv1`, A
  is the im2col of the mel (n_mels × 2·ctx) with kernel 3, stride 1 and
  padding 1, giving (2·ctx) × (3·n_mels) with columns ordered (channel,
  tap). B is `conv1.weight.reshape(d, 3·n_mels).T`. `conv2` is the same with
  stride 2 over the GELU'd conv1 output, giving ctx × 3d. The bias is added on
  the host, then `gelu` runs, then the positional embedding's first ctx rows
  are added. Reference: `im2col`.
- **`gelu`.** One fixed tile for every elementwise GELU. The input is
  flattened, cut into chunks of `rows·cols`, and the last chunk is
  zero-padded.
- **`proj`.** q, k, v, the attention output, and (on the fused MLP path)
  every fc1 column chunk share this design, with the **bias folded**
  (`bias_folded: true`):
  - A gets a column of ones at index `K−1 = d`.
  - B gets the bias as row `d`. k has no bias in Whisper, so a zero row goes
    there.
  - `c` then already holds `x @ W + b`.
  - Sharing one design keeps the encoder within 6 hw_contexts.
- **`qk` → `softmax` → `av`.** The attention chain, all heads per launch, with
  no host round trip in between:
  - `qk`: per head, A is `q_h · hd^-0.25` (Mp × hd) and B is
    `(k_h · hd^-0.25)ᵀ` (hd × Np). Both are stacked per head at the given
    `head_stride_bytes` in one parent BO each. The bf16 output goes straight
    into head h's slice of the softmax input stack, at offset
    `h · Mp·Np·2` (`"into": "softmax.in"`).
  - `softmax`: one launch over `nh·Mp` rows, `buffer_bytes` per stack.
  - `av`: A is head h's slice of the softmax output (`"from": "softmax.out"`)
    and B is `v_h`, both stacked. The f32 output is stacked per head; the
    host concatenates the first `hd` columns of each head.
  - `attn` in the step gives the shared `Mp`/`Np`. Both GEMMs use one row
    padding, `Mp`; see `rawxrt.chain_designs` for why.
- **MLP, `mlp.path`.**
  - `plain` (fc1 N ≤ 4096: tiny, base, small, medium): `fc1` is a plain GEMM
    (bias on the host), then `gelu`, then `fc2`.
  - `fused` (large, where fc1 N = 5120 does not compile as one GEMM): fc1 runs
    as `chunks` column chunks of width d on `proj`, with bias folded. Each
    chunk's f32 result is GELU'd by `fc1_gelu` straight into its band of
    `fc2`'s A:
    - the band is a sub-BO at `j · offset_bytes_per_chunk` of a buffer with
      row stride `row_stride_bytes`;
    - then `fc2` runs from that buffer.
  - fc2's bias is added on the host.
- **Host ops between roles** (layer norms, residual adds, bias adds, im2col,
  slicing) follow `encoder()` exactly. `tests/data/packs/*/fixtures` pins
  them down; see "Fixtures".

## Ladder selection

The encoder runs on a context of `ctx` positions (50 per second of audio), and
whisper.cpp must use the same value as its `audio_ctx`, so that the mel window
and the cross-attention K/V match:

```
need = ceil(secs · positions_per_sec · (1 + margin_ratio)) + margin_positions
ctx  = the smallest ladder ctx >= need; the full n_audio_ctx from 30 s on
```

This is the rule `serve/` uses (`NPU_STT_CTX_FROM_AUDIO`). `max_audio_secs`
is that rule solved for each step. The encoder callback declines any `n_ctx`
that is not exactly a ladder step, and whisper.cpp then runs its own encoder.
On the NPU path this replaces voxtype's `calculate_audio_ctx` heuristic.
Measured with whisper.cpp itself running at `audio_ctx` = ladder ctx (512,
768, 1280 and 1500): the text is identical to the service's ct2 decoder on the
same NPU features (`prototype/whisper-cpp-hook/README.md`).

## Memory

XRT maps BOs with `MAP_LOCKED`, and the kernel checks **each mapping** against
`RLIMIT_MEMLOCK`. It does not check the total (`VmLck` stays 0). So the limit
has to admit the largest single mapping:

```
memlock_min_bytes = next power of two >= max(heap_bytes, max_bo_bytes)
```

`max_bo_bytes` is computed over every BO the roles allocate. At ctx 1500 for
large-v3-turbo it is the attention stack, `nh · Mp · Np · 2` = 94,371,840 B,
which is exactly the size of the mapping that fails at 64 MiB. That gives
128 MiB for turbo and 64 MiB for tiny. With `CAP_IPC_LOCK`, any limit works.
Below the floor, the first BO fails with `EAGAIN`, or worse, the large
attention BO fails at the top ladder step only. A loader should allocate the
top step's buffers at startup, so that a limit that is too low fails there and
not mid-dictation.

## Fixtures

`tests/data/packs/tiny-*/fixtures/ctx512/` holds one encoder pass of
whisper-tiny on `tests/data/jfk.wav` (the first 10.24 s) at ctx 512, run on
the NPU with weights from `ggml-tiny.bin`. All files are float32 `.npy`:

| file | shape | what |
|---|---|---|
| `mel.npy` | 80 × 1024 | the encoder input |
| `conv1_gelu.npy` | 1024 × 384 | after conv1 + bias + GELU |
| `block0_in.npy` | 512 × 384 | after conv2 + bias + GELU + positional embedding |
| `block1_in.npy` | 512 × 384 | block 0's output |
| `out.npy` | 512 × 384 | after `ln_post`: the callback's output |

They let the host side (im2col, padding, bands and strides, layer norm,
residuals) be unit-tested without an NPU. Compare with a tolerance, not
bit-exactly: the device rounds to bf16, and a host reimplementation of the
GELU tile or of a layer norm may differ in the last bits. The NPU path itself
needs a hardware-gated integration test against `out.npy`.
