# Validação dos impedimentos citados pelo upstream

O README e o HISTORY.md do upstream (`drakosha/whisper-xdna`, verificados por
ele em 07/2026, num Ryzen 7 PRO 8845HS) apontam limites e caminhos fechados. O
que o upstream investigou não é necessariamente verdade hoje nem neste host,
então cada impedimento foi tratado como hipótese e revalidado com pesquisa ou
medição própria em 2026-10-17 (Ryzen 7 8700G, XDNA1, firmware 1.5.5.391,
kernel 7.2.5, `large-v3-turbo`).

Vereditos:
- **Confirmado:** a medição própria sustenta a alegação.
- **Corrigido:** a conclusão se mantém, mas os números ou o motivo estavam
  errados.
- **Refutado:** a medição própria contradiz a alegação.
- **Desatualizado:** era verdade quando foi escrito e mudou.
- **Não revalidado:** fora do escopo desta rodada; mantido como hipótese.

| # | Alegação do upstream | Veredito | Resumo |
|---|---|---|---|
| U1 | O decoder do turbo é pequeno: "~63 ms of a 2234 ms recognition" | **Refutado** | 660–1564 ms aqui (PyTorch fp32), até 1/3 do total. |
| U2 | "~815 M weights read per token, 1.6 GB in bf16", "32 × 1.2 ms" | **Corrigido** | Números do large-v3. O turbo tem 4 camadas e 172 M parâmetros (344 MB/token em bf16). |
| U3 | O decoder é a forma errada para a NPU: limitado por memória | **Confirmado (com números novos)** | GEMV na NPU: 3,5–5,4 GB/s por core, contra ~33–48 GB/s efetivos da CPU. |
| U4 | Custo de ~1,1–1,3 ms por chamada | **Corrigido** | É o custo do wrapper IRON; o próprio upstream mediu 0,267 ms em pyxrt direto. Medido aqui: ~0,1–0,2 ms de overhead por chamada. Não é o limitante. |
| U5 | whisper.cpp da AMD: NPU "Windows only" | **Desatualizado** | O whisper.cpp v1.9.4 oficial tem VitisAI com **Linux (Ubuntu 24.04)**, mas segue só XDNA2 (`deviceName = "stx"`, Ryzen AI 300/400). |
| U6 | RyzenAI-SW ASR demo exige stack com conta (SDK + conda + VitisAI EP) | **Não revalidado** | O Ryzen AI ganhou instalador Linux na 1.7.1, mas nada mostra o demo ASR em XDNA1 no Linux. Um guia no 8700G viu LLMs falharem por falta de overlay 4x4. |
| U7 | Checkpoints `magicunicorn/*-amd-npu-int8` sem pesos | **Confirmado** | Hoje os repositórios respondem 401. |
| U8 | KV cache cresce e o MLIR-AIE compila formas estáticas | **Confirmado em princípio** | Daria para contornar com padding até 448 tokens, mas U3 decide antes. |
| U9 | `NPU_STT_TORCH_THREADS=16` como padrão | **Corrigido (implícito)** | No 8700G (8 núcleos/16 threads), 8 threads é 6–10% mais rápido em todas as variantes. |
| U10 | Sobrepor host e NPU rende 1,4%; dividir GEMM CPU+NPU rende 64% do aditivo; limite de 6 `hw_context`; `AIE_LOOP_UNROLL_FULL` −14%; flash attention mais lenta | **Não revalidado** | Fora do escopo do decoder. |
| U11 | int8 no encoder: velocidade 1,51×, transcrição 21/27 idêntica (reaberto) | **Não revalidado** | É o próximo alvo, já que o encoder passa a ser o gargalo ([decoder.md](decoder.md#projeção-do-caminho-npu-com-o-decoder-trocado)). |

## Detalhes

### U1. Tempo do decoder do turbo

Medido com o `serve/` (`X-NPU-Dec-Ms`) e com
[`bench/probe_decoder_cpu.py`](../bench/probe_decoder_cpu.py), na mesma
configuração do serviço (PyTorch fp32, `whisper.decode`, greedy):

| Clipe | Tokens | Serviço (16 threads) | Probe, 16 threads | Probe, 8 threads |
|---|---|---|---|---|
| 6 s | 28 | 607–684 ms | 726 ms | 669 ms |
| 20 s | 64 | 1429–1465 ms | 1564 ms | 1434 ms |

São ~21–22 ms/token, longe de "63 ms" para uma requisição inteira. O upstream
não detalha como mediu os 63 ms. Uma hipótese compatível é um clipe com
pouquíssimos tokens: a fala em russo do 60 s quase em silêncio decodifica para
`'и'`, como ele mesmo registra. Para ditado real em pt-BR, o decoder pesa de
1/3 a 1/4 do tempo.

### U2. Tamanho do decoder

O probe imprime `decoder params 172 M` para o `large-v3-turbo` (4 camadas,
d=1280: ~105 M nas camadas + 66 M no embedding/logits amarrado). Os ~815 M e as
"32 camadas" do README são do `large-v3`. Para o turbo:
- bytes por token: 688 MB em fp32, 344 MB em bf16/f16, ~172 MB em int8;
- GEMVs por token: 4 camadas × 8 (q, k, v, o da self-attention; q e o da
  cross-attention; fc1, fc2), mais os logits: ~33, não "32 × 1,2 ms".

### U3. Decoder na NPU é limitado por memória

Medido na NPU deste host com o design `matrix_vector` do mlir-aie 1.3.4 (o da
imagem), com o serviço `stt` parado durante o teste para não disputar
contextos:

| GEMV (M×K) | Pesos | Tempo NPU (méd. / mín.) | Banda | Ponta a ponta (méd.) |
|---|---|---|---|---|
| 1280×1280 (q/k/v/o) | 3,2 MB | 931 / 682 µs | 3,5–4,8 GB/s | 1144 µs |
| 1280×5120 (fc2) | 12,8 MB | 2490 / 2365 µs | 5,1–5,4 GB/s | 2696 µs |
| 5120×1280 (fc1) | 12,8 MB | — | — | **não compila** ("NPU lowering pipeline failed") |

- O design é de **um único core** ("multi-core extension is left for a future
  revision") e int16. A variante bf16 do kernel `mv.cc` está comentada na
  1.3.4. Como int16 e bf16 têm 2 bytes, a banda vale como proxy.
- A 344 MB/token, um core daria **~64 ms/token**, contra 5–10 ms/token na CPU.
  Mesmo um design com 4 colunas e escala perfeita (~20 GB/s) não alcançaria a
  CPU (33–48 GB/s efetivos), e ainda disputaria a mesma DDR.
- O design `whole_array` usado no encoder arredonda M para múltiplos de 512
  (`padded_shape`): um GEMV com M = 1 vira uma GEMM de 512 linhas.
- Conclusão: o impedimento **se mantém**, pela banda de DMA da NPU e não pelo
  custo por chamada (U4). Levar o decoder para a NPU só faria sentido com
  batching ou decodificação especulativa, como o upstream já notava.

Reprodução:

```bash
docker compose stop stt
docker run --rm --device /dev/accel/accel0 --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add IPC_LOCK --ulimit memlock=-1 --entrypoint bash kostryukov/npu-whisper:1.3.4 -c '
  cd /opt/mlir-aie/programming_examples/basic/matrix_multiplication/matrix_vector &&
  python3 matrix_vector.py -M 1280 -K 1280 -m 32 -k 32 --iters 50 --warmup 5'
docker compose start stt
```

### U4. Custo por chamada

- O próprio README do upstream, na seção de achados: "The IRON Python wrapper
  costs ~1.1 ms per call. The same 512³ GEMM is 1.385 ms through IRON and
  **0.267 ms through pyxrt directly**". Ele cita ainda `pyxrt.runlist` 24%
  melhor e encadeamento sem custo.
- No GEMV acima, a diferença entre o tempo de ponta a ponta e o tempo de NPU
  é de ~0,1–0,2 ms por chamada.
- Com ~33 chamadas por token, isso dá ~3–7 ms/token de overhead (menos com
  runlist e encadeamento): relevante, mas abaixo do custo de banda (U3).

### U5. whisper.cpp com NPU AMD

- O README do whisper.cpp **v1.9.4** (ggml-org, 2026-09-11) inclui "AMD Ryzen AI
  NPU support", com plataformas suportadas **Windows 11 e Linux (Ubuntu 24.04
  LTS)**, via XRT + FlexML runtime.
- O escopo é "AMD Ryzen AI 300 and 400 Series", e
  `src/vitisai/whisper-vitisai-encoder.cpp` fixa `options.deviceName = "stx"`
  (Strix). Os caches de encoder vêm da coleção
  `amd/ryzen-ai-whisper-npu-optimized-onnx-models` (tiny, base, small,
  small-en, medium, large-v3, large-turbo).
- Conclusão: o "Windows only" ficou desatualizado; o "não é XDNA1" continua
  valendo. O binário compilado aqui mostra `WHISPER : VITISAI = 0`.

### U9. Threads

Em todas as variantes do decoder (PyTorch fp32/bf16/int8 e CTranslate2 int8),
8 threads foi 6–10% mais rápido que 16. Isso é coerente com uma carga
limitada por memória num CPU de 8 núcleos com SMT. Ver
[decoder.md](decoder.md#onde-o-tempo-vai).
