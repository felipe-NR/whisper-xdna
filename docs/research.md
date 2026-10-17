# Pesquisa: ditado local (pt-BR/en) usando a NPU XDNA1 do Ryzen 7 8700G

Registro das descobertas feitas em 2026-10-17 ao avaliar alternativas ao
[Voxtype](https://voxtype.io/) que usem a NPU do host, e do plano de comparação
NPU × iGPU.

## Host

| Item | Valor |
|---|---|
| CPU/APU | AMD Ryzen 7 8700G (Phoenix/Hawk Point, desktop) |
| NPU | XDNA1, PCI `1022:1502` "AMD IPU Device", BDF `0000:67:00.1`, ~16 TOPS |
| iGPU | Radeon 780M (Phoenix1), Vulkan disponível |
| SO | Omarchy (Arch Linux), kernel `7.2.5-3-omarchy` |
| Driver NPU | `amdxdna` in-tree, `amdxdna_accel_driver 0.10.0`, `/dev/accel/accel0` |
| Firmware NPU | `1.5.5.391` (`amdnpu/1502_00/npu_7.sbin`) |
| XRT no host | não instalado (o stack de userspace vem na imagem Docker) |
| memlock do usuário | 8192 KiB (o container define o seu próprio: `ulimits.memlock: -1`) |
| RAM / CPUs | 58 GiB / 16 threads |

## Voxtype (ponto de partida)

- Versão `1.1.1`, instalado via `omarchy voxtype install` (pacote `voxtype-bin`).
- Configuração padrão do Omarchy: `model = "base.en"`, `language = "en"`. O
  modelo `.en` só reconhece inglês e ignora `language`.
- Para pt-BR: usar um modelo multilíngue (`small`, `large-v3-turbo`…) e
  `language = ["pt", "en"]`. Arrays de idiomas são suportados desde que o modo
  seja `local` ("Multi-language allowlists … can be set in config.toml as an
  array"; no modo `remote` só o primeiro idioma é usado). A detecção automática
  fica restrita aos idiomas da lista (~50 ms na primeira janela). Não existe
  uma opção de idioma por gravação em `voxtype record`.
- Aceleração disponível no voxtype:
  - `voxtype setup gpu`: Vulkan (Whisper) ou CUDA/MIGraphX (Parakeet).
  - `voxtype setup npu`: **somente NPU Intel via OpenVINO**. A NPU AMD XDNA não
    é suportada.
- Estado encontrado após a instalação: backend `CPU (AVX-512)` ativo, apesar de
  o `GPU (Vulkan)` estar instalado. O `voxtype setup gpu --enable` do
  instalador do Omarchy falhou em silêncio (`|| true`) porque precisa de root
  para trocar o `/usr/bin/voxtype`.
- Integração possível com um servidor externo: `whisper.mode = "remote"` +
  `remote_endpoint`, que espera a API **compatível com OpenAI**
  (`/v1/audio/transcriptions`). Também há `mode = "cli"` (`whisper_cli_path`).

## Alternativas avaliadas para a NPU

### Descartadas (verificadas nas fontes)

| Alegação | Verificação | Fonte |
|---|---|---|
| FastFlowLM roda Whisper no XDNA1 | Falso. README: "supports all Ryzen AI Series chips with **XDNA2** NPUs (Strix, Strix Halo, Kraken, Gorgon Point)". Linux suportado desde 03/2026. | [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) |
| Lemonade Server faz Whisper na NPU no Linux | Só XDNA2. A tabela de backends do repositório lista `whispercpp` na NPU **somente no Windows**; no Linux usa Vulkan ou CPU. | [Lemonade FLM Linux](https://lemonade-server.ai/flm_npu_linux.html) |
| Artigo do dev.to prova Whisper na NPU no Linux | Verdadeiro, mas num Ryzen AI 9 365 (**Strix Point, XDNA2**) com FastFlowLM. Não testado em XDNA1. | [dev.to/jac-76](https://dev.to/jac-76/running-whisper-llms-on-an-amd-npu-under-linux-2o1h) |
| whisper.cpp da AMD (Ryzen AI) com encoder na NPU | Existe, mas é documentado para Ryzen AI 300 e: "NPU acceleration is currently supported on **Windows only**, with Linux support planned." | [Ryzen AI – Whisper.cpp](https://ryzenai.docs.amd.com/en/latest/whisper_cpp.html), [amd/whisper.cpp](https://github.com/amd/whisper.cpp) |
| `amd/NPU-Whisper-Base-Small` | Só Windows (Ryzen AI 1.5, VS 2022). | [HF](https://huggingface.co/amd/NPU-Whisper-Base-Small) |
| `magicunicorn/whisper-{small,medium}-amd-npu-int8` (MLIR-AIE2, 5200 tok/s, 8–10 W, Ubuntu 22.04) | **Não existem publicamente**: HTTP 401 na página e na API. Os 11 modelos públicos do autor não incluem Whisper para NPU AMD. O `whisper-base-int8` dele é OpenVINO para iGPU **Intel**, sem código e com 9 downloads. O projeto "Unicorn Amanuensis" citado dá 404. | [HF API](https://huggingface.co/api/models?author=magicunicorn) |
| Ryzen AI Software no Linux para Phoenix/Hawk Point | O 1.8.0 lista Phoenix e Hawk Point e tem instalador Linux. Porém um guia testado **num 8700G** mostra que só o básico do XRT funciona (vadd ~69 µs) e que LLMs falham por falta de overlay de batched-matmul para o alvo 4x4 (Phoenix). Whisper não é coberto. | [Release notes 1.8.0](https://ryzenai.docs.amd.com/en/latest/relnotes.html), [gist 1kaiser](https://gist.github.com/1kaiser/f5abe5f9c91e783b4d759266f867653e) |
| IRON / mlir-aie | Toolchain de baixo nível com suporte a NPU1 (Phoenix) e operadores (GEMM, MHA, RMSNorm…). Não há exemplo de Whisper pronto; é a base sobre a qual o whisper-xdna foi construído. | [amd/IRON](https://github.com/amd/iron), [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) |

Observações do guia do 8700G que podem ser úteis aqui:
- Se aparecer `SVA bind device failed (ret -95)`, a saída é o parâmetro
  `force_iova=1` do módulo `amdxdna` (via `modprobe.d`).
- O limite de memlock de 8 MB vem do `DefaultLimitMEMLOCK` do systemd.
  `limits.conf` sozinho não resolve.

### Viável: whisper-xdna (este repositório)

- Encoder do Whisper inteiro na NPU XDNA1, com todas as multiplicações de
  matriz no array AIE. O decoder roda na CPU (PyTorch/openai-whisper). Usa
  apenas toolchain aberto: XRT 2.21.75 (Debian trixie-backports), MLIR-AIE/IRON
  v1.3.4 e peano.
- Testado pelo autor num Ryzen 7 PRO 8845HS (Hawk Point, XDNA1 com 4 colunas),
  kernel 7.0.13, driver 0.6.0, **firmware 1.5.5.391**, igual ao deste host.
- Números do autor:
  - Encoder `large-v3` em ctx=1500: **3344 ms na NPU** contra 6940 ms do
    whisper.cpp q5_0 na CPU, usando 0,25 núcleo contra 15,3.
  - `large-v3-turbo` com um clipe de 11 s em ctx=1500: 3346 ms.
- Serviço HTTP (`serve/`): `POST /inference` no formato do whisper.cpp
  (`file`, `language`, `audio_ctx`, `response_format`), porta 8090. O idioma é
  **fixo por requisição** (padrão `NPU_STT_LANG=ru`), sem auto-detecção. Não
  existe `/v1/audio/transcriptions`.
- Maturidade: 3 commits, sem releases, precisão checada em 3 clipes (en/ru).
  Português não testado.
- Requisitos: Docker, ~12 GB de imagem (`WITH_REFERENCE=1` para o serviço),
  8–9 GiB de RAM, `IPC_LOCK` e `memlock` ilimitado no container. Na primeira
  execução baixa o checkpoint (~1,6 GB) e compila os overlays (minutos).

Projetos derivados:
- [npusubs / bazarr-subtitles-xdna1](https://github.com/Joassens/bazarr-subtitles-xdna1):
  usa o whisper-xdna e acrescenta `POST /v1/audio/transcriptions` compatível com
  OpenAI, VAD e diarização. Medido no 8845HS: ~4 s por janela de 30 s no
  encoder; 4 min de áudio levaram ~89 s com greedy (RTF ≈ 0,37). O autor diz:
  "The NPU path is **slower** than a GPU." Status: "early", sem autenticação.
- [npu-whisper (jac-76)](https://github.com/jac-76/npu-whisper): wrapper do
  FastFlowLM, portanto só XDNA2.

## Expectativa para ditado

| | Latência esperada (frase curta) | CPU/GPU livres | Maturidade |
|---|---|---|---|
| iGPU 780M, Vulkan (voxtype local) | ~1–2 s (estimativa a medir) | GPU ocupada por instantes | estável |
| NPU XDNA1 (whisper-xdna) | ~4–6 s (encoder ~3,3 s em ctx=1500 + decoder na CPU; contextos menores na escada `NPU_STT_LADDER` devem reduzir isso) | sim | protótipo |

## Montagem do ambiente de teste

- Voxtype: `sudo voxtype setup gpu --enable` e depois
  `systemctl --user restart voxtype`. O daemon carrega o modelo em
  `AMD Radeon 780M Graphics (RADV PHOENIX)` (`uma: 1`, `fp16: 1`,
  `matrix cores: KHR_coopmat`). O binário ativo vira
  `/usr/lib/voxtype/voxtype-vulkan`, e o CLI `voxtype transcribe` usa o mesmo.
- NPU: em vez do build local (~12 GB), usei a imagem publicada
  `kostryukov/npu-whisper:1.3.4` (~1 GB comprimido, de 2026-07-26). Ela já
  inclui o `refenv` (torch `2.13.0+cpu` + openai-whisper) e o ffmpeg.

  ```bash
  NPU_WHISPER_IMAGE=kostryukov/npu-whisper:1.3.4 RENDER_GID=987 \
  NPU_STT_LANG=pt STT_PORT=127.0.0.1:8090 docker compose up -d stt
  ```

  - **`STT_PORT=127.0.0.1:8090` é importante.** O padrão do compose publica em
    `0.0.0.0:8090`, e o serviço não tem autenticação.
  - Na primeira subida: o checkpoint `large-v3-turbo.pt` foi baixado de novo
    (aviso de SHA256 divergente no volume recém-criado, 30 s), e o prewarm
    compilou os overlays da escada `512,768,1024,1280,1500` em ~85 s (de 11 a
    21 s por contexto), mais 2,3 s do decoder. As subidas seguintes reaproveitam
    os volumes `aot` e `models`.
  - Nada de `force_iova` nem de memlock no host foi necessário. O container
    roda com `IPC_LOCK` e `memlock: -1`, e o `/dev/accel/accel0` do host já é
    `crw-rw-rw- root:render`.
  - Docker sem relogin: `sudo setfacl -m u:$USER:rw /var/run/docker.sock`
    (temporário, até o Docker reiniciar; depois vale o grupo `docker`).

## Comparação prática

Script: [`tools/compare_dictation.py`](../tools/compare_dictation.py), só com
a stdlib. Ele roda os dois caminhos sobre o mesmo WAV (16 kHz, mono), faz um
aquecimento e mais N repetições, e calcula o WER contra um `.txt` de
referência ao lado do áudio:

- `gpu`: `voxtype -v --model large-v3-turbo --language pt transcribe <wav>`.
  Conta só a inferência que o voxtype registra ("Transcription completed in");
  a carga do modelo (~340 ms) fica de fora, porque o daemon mantém o modelo
  residente. O texto vem da última linha do stdout, já que a linha de log
  trunca transcrições longas com "...".
- `npu@<ctx>`: `POST /inference` com `language=pt`. Os tempos vêm dos headers
  `X-NPU-*`. `auto` dimensiona o `audio_ctx` ao clipe
  (`ceil(s·50·1,1)+16`) e o servidor arredonda para cima na escada.

```bash
python3 -I tools/compare_dictation.py --runs 3 --json results.json pt-curto.wav pt-longo.wav
```

### Áudios

Gravados pelo usuário com `pw-record --rate 16000 --channels 1 --format s16`.
Os áudios não estão versionados; os textos de referência estão abaixo.

- `pt-curto.wav` (5,95 s): "Amanhã de manhã preciso revisar o relatório e
  enviar o e-mail para a equipe de infraestrutura."
- `pt-longo.wav` (19,94 s): "O processador Ryzen sete oito mil e setecentos G
  possui uma unidade de processamento neural integrada. Neste teste, vamos
  comparar a transcrição de voz usando a placa de vídeo integrada e a NPU,
  medindo o tempo de resposta e a qualidade do texto em português do Brasil."

**Qualidade do microfone:** há um zumbido constante de **60 Hz** (rede
elétrica) em ~-19 dBFS, com a mesma energia da fala (SNR de banda larga
≈ 0,6 dB). Na faixa da voz (passa-alta em 200 Hz), ruído em ~-48 dBFS e fala
em ~-29 dBFS dão uma **SNR ≈ 18–19 dB**, aceitável. A voz está baixa. Como os
dois caminhos recebem o mesmo áudio, a comparação entre eles é justa, mas o
WER absoluto pode estar pior que o normal.

### Resultados (2026-10-17, 3 repetições após o aquecimento, `large-v3-turbo`, `pt`)

| Áudio | Caminho | Inferência (mediana) | RTF | Detalhe | WER* |
|---|---|---|---|---|---|
| curto (6,0 s) | **gpu (Vulkan 780M)** | **1440 ms** | 0,24 | modelo residente no daemon | 0,0% |
| curto (6,0 s) | npu@auto | 1804 ms | 0,30 | ctx=512: enc 1192 + dec 615 ms | 0,0% |
| curto (6,0 s) | npu@1500 | 3815 ms | 0,64 | ctx=1500: enc 3148 + dec 684 ms | 0,0% |
| longo (19,9 s) | **gpu (Vulkan 780M)** | **1710 ms** | 0,09 | | 12,8% |
| longo (19,9 s) | npu@auto | 4544 ms | 0,23 | ctx=1280: enc 3101 + dec 1429 ms | 12,8% |
| longo (19,9 s) | npu@1500 | 4630 ms | 0,23 | ctx=1500: enc 3162 + dec 1465 ms | 12,8% |

\* **As transcrições foram idênticas palavra por palavra nos dois caminhos.**
Os 12,8% do áudio longo vêm quase todos da referência ("sete oito mil e
setecentos G" por extenso, contra "7 8600G" em dígitos). O único erro de
conteúdo é **8600G no lugar de 8700G**, igual nos dois caminhos.

Uso de CPU por transcrição (tempo de CPU ÷ tempo de parede):

| Áudio | gpu (voxtype, inclui a carga do modelo) | npu (cgroup do container) |
|---|---|---|
| curto | 0,66 s de CPU, ~0,4 núcleo | 2,21 s de CPU, ~0,8 núcleo |
| longo | 0,80 s de CPU, ~0,4 núcleo | 4,44 s de CPU, ~0,8 núcleo |

Outras observações:
- O tempo da NPU é dominado pelo encoder (~3,1 s em ctx ≥ 1280), que cresce
  com o contexto e não com a fala; ctx=512 cobre ~10 s de áudio. O decoder em
  PyTorch fp32 na CPU soma de 0,6 a 1,5 s.
- Trocar de contexto entre requisições custa ~1 s a mais (desmonta e monta a
  geometria do hardware; o primeiro pedido curto após um de 1500 levou 2,85 s).
- A NPU **não** libera CPU no ditado: o decoder do whisper-xdna usa mais CPU
  que o caminho inteiro do whisper.cpp na iGPU.
- Consumo de energia não foi medido (o RAPL exige root para ler o
  `energy_uj`).

### Conclusões

1. **Para ditado interativo, a iGPU vence.** É de 1,25× (frase curta) a 2,7×
   (20 s) mais rápida, usa metade da CPU e é estável. Fica como o padrão do
   voxtype.
2. **A NPU XDNA1 funciona de verdade neste host** com o whisper-xdna (firmware
   1.5.5.391, kernel 7.2.5) e entrega a mesma qualidade, mas a latência sobe
   para 1,8–4,6 s, sem ganho de CPU. Faz sentido como experimento, ou para
   transcrição em background quando a GPU estiver ocupada (jogos, LLM local).
3. Os caminhos para melhorar a NPU seriam: um `audio_ctx` dinâmico no cliente
   (já ajuda nas frases curtas); o decoder em int8 ou no whisper.cpp
   (`bench/probe_int8_decode.py`, ver HISTORY.md); e evitar a troca de
   contexto fixando um ctx único para o ditado.
4. O microfone tem um zumbido de 60 Hz (provável laço de terra ou cabo sem
   blindagem) que precisa ser corrigido no hardware, ou com um filtro
   passa-alta/notch no PipeWire, antes de ajustar VAD e sensibilidade.

## Pendências

- [x] Ativar o Vulkan no voxtype (`sudo voxtype setup gpu --enable`).
- [x] Ativar o Docker e dar acesso ao usuário.
- [x] Subir o `stt` com `NPU_STT_LANG=pt` (imagem publicada, só em localhost).
- [x] Medir latência, qualidade e CPU nos dois caminhos.
- [ ] Integrar com o voxtype: um shim `/v1/audio/transcriptions` →
      `/inference` para usar `whisper.mode = "remote"`. A prioridade é baixa,
      dados os resultados.
- [ ] Medir energia (RAPL com root) no ditado: GPU × NPU.
- [ ] Repetir com o microfone sem o zumbido de 60 Hz e com mais frases.
- [x] Fork no GitHub: <https://github.com/felipe-NR/whisper-xdna> (`origin`),
      com `upstream` apontando para `drakosha/whisper-xdna`.
