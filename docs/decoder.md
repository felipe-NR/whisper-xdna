# Decoder na CPU: auditoria e plano de refatoração

Análise de 2026-10-17 sobre a pergunta "dá para refatorar o decoder do
whisper-xdna para usar instruções mais eficientes?". As alegações recebidas
foram verificadas na fonte ou medidas neste host (Ryzen 7 8700G, 8 núcleos
Zen 4 / 16 threads, `large-v3-turbo`, áudios pt-BR de
[research.md](research.md#áudios)). A validação dos impedimentos citados pelo
upstream está em [upstream-validation.md](upstream-validation.md).

## Resumo

- O decoder do serviço (`serve/npu_stt_server.py`) é o `whisper.decode` do
  openai-whisper em **PyTorch fp32** (`DecodingOptions(fp16=False)`), com
  **16 threads**. Ele **já usa AVX-512**: `torch.backends.cpu.get_cpu_capability()
  == "AVX512"`, com MKL 2024.2 e oneDNN 3.12. O problema não é falta de SIMD,
  é ler 4 bytes por peso numa carga limitada por banda de memória.
- No `large-v3-turbo`, o decoder (4 camadas, **172 M parâmetros**) não é um
  detalhe: custa **660–1564 ms** por requisição, até 1/3 do tempo total do
  caminho NPU.
- Reduzir bytes por peso resolve sem escrever intrínsecos. Com o mesmo
  encoder output e texto idêntico nos dois clipes:

  | Decoder na CPU | ms/token | curto / longo | vs. atual |
  |---|---|---|---|
  | PyTorch fp32, 16 threads (**atual**) | 22,1 | 726 / 1564 ms | 1× |
  | PyTorch fp32, 8 threads | 20,7 | 660 / 1397 ms | 1,1× |
  | PyTorch bf16 (oneDNN, `avx512_bf16`), 8 threads | 10,2 | 321 / 687 ms | 2,3× |
  | PyTorch int8 dinâmico (fbgemm, VNNI), 8 threads | 12,6 | 395 / 839 ms | 1,8× |
  | whisper.cpp v1.9.4, ggml f16 (AVX-512), 8 threads | ~7 | 200 / 435 ms | 3,6× |
  | CTranslate2 4.8.2 fp32, 8 threads | 13,2–14,6 | 408 / 844 ms | 1,8× |
  | **CTranslate2 4.8.2 int8, 8 threads** | **5,2–5,9** | **165 / 335 ms** | **4,4–4,7×** |

- **Recomendação:** trocar o decoder por CTranslate2 int8 com 8 threads. O
  `generate()` do CTranslate2 aceita a saída do encoder diretamente, então o
  encoder continua na NPU e a mudança fica no Python do serviço. Não precisa
  portar nada para C++ nem escrever SIMD à mão.
- Isso **não** leva o decoder para a NPU. No XDNA1, um GEMV na NPU fica de 7 a
  10× abaixo da banda que a CPU já alcança
  ([upstream-validation.md](upstream-validation.md#u3-decoder-na-npu-é-limitado-por-memória)).

## Auditoria do texto recebido

| # | Alegação | Veredito | Evidência |
|---|---|---|---|
| 1 | Ryzen AI SW roda Whisper com `python -m amd_whisper --npu` | **Falso (comando)** | O README do [RyzenAI-SW ASR demo](https://github.com/amd/RyzenAI-SW/blob/main/Demos/ASR/Whisper/README.md) usa `python run_whisper.py --model-type <tipo> --device npu --input …`. O README não cita hardware nem SO, e nada mostra suporte a XDNA1 no Linux (ver [research.md](research.md#alternativas-avaliadas-para-a-npu)). |
| 2 | `magicunicorn/whisper-medium-amd-npu-int8` são pesos prontos | **Falso** | HTTP 401 na página e na API. O README do upstream já registrava em 07/2026 que esses repositórios não tinham pesos. |
| 2b | `amd/NPU-Whisper-Base-Small` pronto para uso | **Só Windows** | Ryzen AI 1.5 + VS 2022 ([research.md](research.md)). |
| 3 | FastFlowLM roda Whisper (`flm serve --asr`) no XDNA | **Só XDNA2** | README: "XDNA2 NPUs (Strix, Strix Halo, Kraken, Gorgon Point)". Os números de energia citados não se aplicam ao 8700G. |
| 4 | whisper-xdna roda no XDNA1 com toolchain aberto | **Verdadeiro** | Medido neste host ([research.md](research.md#resultados-2026-10-17-3-repetições-após-o-aquecimento-large-v3-turbo-pt)). |
| 5 | memlock de 8 MB quebra; usar `--ulimit memlock=-1` | **Verdadeiro** | `compose.yml` (`IPC_LOCK`, `memlock: -1`); o host tem 8192 KiB. |
| 6 | Vulkan na iGPU pode ser mais rápido que NPU XDNA1 | **Verdadeiro** | Na iGPU: 1,44 s / 1,71 s. Na NPU: 1,80 s / 4,54 s. |
| 7 | "Instruções otimizadas de CPU são altamente utilizadas" | **Verdadeiro, mas não pelo motivo dado** | O PyTorch do serviço já despacha AVX-512 (MKL/oneDNN). O ganho possível vem de **precisão menor** (bf16/int8), não de "ligar SIMD". |
| 8 | "O decoder geralmente roda em ONNX Runtime ou whisper.cpp, não PyTorch" | **Falso para o whisper-xdna** | `serve/npu_stt_server.py` chama `WHISPER.decode(MODEL_OBJ, …)` com `fp16=False`. Vale para o demo da AMD (ONNX + VitisAI EP), que não roda neste host. |
| 9 | No demo da AMD, ~93% dos operadores do decoder vão para a NPU em BFP16 | **Verdadeiro, com escopo** | README: Whisper-**Base**, decoder 93,4% NPU (341 VAIML + 24 CPU), BFP16. Não há dado para turbo nem para XDNA1. |
| 10 | "Mesmo no whisper-xdna o decoder opera em FP16/BF16/INT8" | **Falso** | fp32: `DecodingOptions(fp16=False)` e `HOLDER["feats"] = …float()`. |
| 11 | PyTorch fp32 = "matemática escalar FP32 simples" | **Falso** | `cpu_capability=AVX512` e banda efetiva medida de ~33 GB/s (688 MB/token em 20,7 ms), coisa que código escalar não atinge. |
| 12 | Zen 4 tem AVX-512 e FMA | **Verdadeiro** | `/proc/cpuinfo`: `avx2 fma avx512f avx512bw avx512vl avx512_bf16 avx512_vnni`. |
| 13 | Precisão reduzida derruba a latência por token | **Verdadeiro** | bf16 2,0× e int8 (CTranslate2) 3,9× por token, com texto idêntico. |
| 14 | "Abordagem 1": encoder XDNA dentro do whisper.cpp, decoder ggml | **Viável, mas não é a melhor relação custo/benefício** | O whisper.cpp já tem ganchos de encoder externo (VitisAI, OpenVINO, CoreML), mas exigiria portar o runtime do encoder (`src/rawxrt.py`, cadeias, rowops) para C++. O decoder ggml mediu 200 / 435 ms, mais lento que o CTranslate2 int8 (165 / 335 ms), que não exige porte. |
| 15 | "Abordagem 2": pybind11 + intrínsecos AVX-512 / Highway | **Desnecessária** | A carga é limitada por memória: todas as variantes ficam entre ~33 e 48 GB/s efetivos. O ganho vem de ler menos bytes, e as bibliotecas existentes (oneDNN bf16, CTranslate2 int8) já entregam isso. Um kernel escrito à mão não passa do teto de banda. |
| 16 | Menor consumo de bateria | **Não medido** | Host desktop; o RAPL exige root. |

## Medições

### Onde o tempo vai

- **O overhead do Python é pequeno.** `whisper.decode` fica a 3% de um loop
  greedy mínimo com os mesmos hooks de KV-cache (726 contra 723 ms), então o
  custo é aritmética e memória.
- **É limitado por memória.** Banda efetiva aproximada (pesos lidos por token
  ÷ ms/token):
  - PyTorch fp32: 688 MB / 20,7 ms ≈ 33 GB/s.
  - PyTorch bf16: 344 MB / 10,2 ms ≈ 34 GB/s.
  - ggml f16: 344 MB / ~7 ms ≈ 48 GB/s.
  - CTranslate2 int8: ~172 MB / 5,2 ms ≈ 33 GB/s.

  Cortar bytes por peso corta o tempo quase na mesma proporção.
- **SMT atrapalha.** 8 threads (núcleos físicos) é 6–10% melhor que 16 em
  todas as variantes. O padrão do serviço, `NPU_STT_TORCH_THREADS=16`, é
  subótimo neste CPU.
- **O int8 dinâmico do PyTorch perde para o bf16** porque a projeção de logits
  é o embedding amarrado (51866 × 1280 = 66 M pesos) e não é um `nn.Linear`;
  ela fica em fp32 e domina a leitura. O CTranslate2 quantiza também essa
  projeção, por isso chega a 5,2 ms/token.
- **O primeiro passo** (prompt + K/V de cross-attention sobre 1500 posições)
  custa 75–84 ms em fp32 e 29–31 ms em bf16.

### Projeção do caminho NPU com o decoder trocado

Encoder e mel medidos no serviço; decoder pelos probes acima.

| Clipe | Hoje (enc + dec) | Com CTranslate2 int8 | iGPU (voxtype) |
|---|---|---|---|
| curto, ctx=512 | 1192 + 615 ≈ **1,80 s** | 1192 + 165 ≈ **1,36 s** | 1,44 s |
| longo, ctx=1280 | 3101 + 1435 ≈ **4,54 s** | 3101 + 335 ≈ **3,44 s** | 1,71 s |

Para frases curtas de ditado, o caminho NPU **passaria a empatar ou superar a
iGPU**. Para áudios longos, o gargalo passa a ser só o encoder (~3,1 s em
ctx ≥ 1280), e é nele que a próxima rodada de trabalho deveria focar (int8 no
encoder, reaberto em HISTORY.md, e `audio_ctx` dinâmico).

Ressalva: os probes usam a saída do **encoder de referência em torch**, não a
do encoder bf16 na NPU. As diferenças numéricas entre as duas podem mudar
tokens; a integração precisa revalidar a transcrição de ponta a ponta.

### Ferramentas

| Arquivo | O que mede |
|---|---|
| [`bench/probe_decoder_cpu.py`](../bench/probe_decoder_cpu.py) | `whisper.decode` contra um loop greedy em fp32, bf16 e int8 dinâmico, por número de threads; primeiro passo e ms/token |
| [`bench/probe_decoder_ct2.py`](../bench/probe_decoder_ct2.py) | CTranslate2 `generate()` sobre o mesmo encoder output, por `compute_type` |
| [`tools/compare_dictation.py`](../tools/compare_dictation.py) | Ponta a ponta: voxtype/Vulkan contra o serviço da NPU |

```bash
# PyTorch: imagem do serviço, container descartável
docker run --rm -v whisper-xdna_models:/root/.cache/whisper \
  -v $PWD/bench:/b:ro -v ~/Projects/npu-bench-audio:/a:ro \
  --entrypoint /opt/refenv/bin/python kostryukov/npu-whisper:1.3.4 \
  -W ignore /b/probe_decoder_cpu.py --threads 16,8 /a/pt-curto.wav /a/pt-longo.wav

# CTranslate2: --no-deps, senão o pip troca o numpy da imagem e quebra o numba
docker run --rm -v whisper-xdna_models:/root/.cache/whisper \
  -v $PWD/bench:/b:ro -v ~/Projects/bench-tools/ct2-turbo:/ct2:ro \
  -v ~/Projects/npu-bench-audio:/a:ro --entrypoint bash kostryukov/npu-whisper:1.3.4 -c '
  /opt/refenv/bin/python -m pip install -q --no-deps --target /tmp/ct2 ctranslate2==4.8.2 &&
  PYTHONPATH=/tmp/ct2 /opt/refenv/bin/python -W ignore /b/probe_decoder_ct2.py \
    --ct2 /ct2 --compute-types float32,int8 --threads 8 /a/pt-curto.wav /a/pt-longo.wav'
```

- Modelo CTranslate2 usado: `deepdml/faster-whisper-large-v3-turbo-ct2`
  (192 mil downloads). A wheel de CPU do CTranslate2 4.8.2 oferece só
  `float32`, `int8` e `int8_float32`; `bfloat16` é recusado ("target device or
  backend do not support efficient bfloat16 computation"), apesar do
  `avx512_bf16`.
- whisper.cpp v1.9.4 compilado num container `debian:13` com
  `-DGGML_NATIVE=ON`. O `system_info` mostra `AVX512 = 1 | AVX512_VNNI = 1 |
  AVX512_BF16 = 1`. Greedy: `whisper-cli -bs 1 -bo 1 -t 8 -l pt`. Os passos por
  token entram em `decode time`; o `batchd` é só o prompt.

## Plano de refatoração (proposto, não implementado)

1. **Fase 0, trivial:** `NPU_STT_TORCH_THREADS=8`. Ganho de ~10% no decoder,
   sem tocar em código.
2. **Fase 1, decoder em CTranslate2 int8** em `serve/npu_stt_server.py`:
   - Uma variável `NPU_STT_DECODER=torch|ct2`, com `torch` como padrão para
     manter o comportamento do upstream.
   - Em `_window`: `ct2.generate(StorageView.from_array(feats[None]),
     [sot_sequence_including_notimestamps], beam_size=1)` no lugar de
     `WHISPER.decode`, e o texto via `tokenizer.decode(sequences_ids[0])`.
   - **Mesmo checkpoint.** O upstream alerta que encoder e decoder de
     checkpoints diferentes "confabulam" sem dar erro. Converter o modelo
     CTranslate2 a partir do **mesmo** `large-v3-turbo` (com
     `ct2-transformers-converter --quantization int8`) e registrar o hash, em
     vez de depender de um modelo de terceiros.
   - Sem o decoder torch, o modelo torch só serviria para extrair os pesos do
     encoder no start, e o processo pode soltá-lo: menos RAM residente que os
     ~8–9 GB de hoje.
   - Imagem: `ctranslate2` no `refenv` (ou numa venv nova), fixando a versão do
     numpy compatível com o numba do openai-whisper.
   - Aceitação: `tests/` mais transcrição idêntica (ou diferença justificada)
     contra o caminho torch nos clipes de referência, com o encoder **da NPU**.
3. **Fase 2, opcional:** decoder torch em bf16 como fallback sem dependência
   nova (2,3×), caso o CTranslate2 não entre na imagem.
4. **Fora do escopo do decoder:** o encoder passa a ser o gargalo (int8 nos
   GEMMs, ctx dinâmico no cliente).
