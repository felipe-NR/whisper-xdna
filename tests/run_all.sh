#!/usr/bin/env bash
# Every automated test of this tree, in dependency order, through compose.
#
#   tests/run_all.sh                 # host build cpu npu serve
#   tests/run_all.sh host cpu        # only those stages
#   CLIPS=~/audio tests/run_all.sh serve   # + .wav/.txt pairs (not versioned)
#
# Stages:
#   host   tests/test_compare_dictation.py: stdlib, no container
#   build  the ct2 image (compose.ct2.yml)
#   cpu    test-cpu: serve/ct2_decoder.py unit tests (whisper-tiny) and the
#          large-v3-turbo equivalence run against whisper.decode
#   npu    test-npu: the NPU acceptance gate (rowops, chain accuracy, encoder
#          golden + transcription). stt is stopped first: it holds the device.
#   serve  stt with the torch decoder, then with ct2: tests/test_serve_http.py
#          on each, and the ct2 transcripts must equal the torch ones
#
# stt is put back as .env describes it when the script exits, pass or fail.
set -uo pipefail
cd "$(dirname "$0")/.."

export COMPOSE_FILE=compose.yml:compose.ct2.yml
export RENDER_GID="${RENDER_GID:-$(getent group render | cut -d: -f3)}"
URL="${STT_URL:-http://127.0.0.1:8090}"
STAGES="${*:-host build cpu npu serve}"
TMP="$(mktemp -d)"
FAILED=()
TOUCHED_STT=0

wait_ready() {
  for _ in $(seq 1 120); do
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    [ "$(docker inspect -f '{{.State.Running}}' npu-whisper-stt 2>/dev/null)" = true ] || break
    sleep 3
  done
  echo "stt did not become ready; last log lines:" >&2
  docker logs --tail 20 npu-whisper-stt >&2
  return 1
}

restore() {
  if [ "$TOUCHED_STT" = 1 ]; then
    echo "== restoring stt from .env"
    docker compose up -d stt >/dev/null 2>&1 && wait_ready && echo "   stt ready"
  fi
  rm -rf "$TMP"
}
trap restore EXIT

stage() {      # stage NAME COMMAND...
  local name="$1"; shift
  echo; echo "========== $name"
  if "$@"; then echo "---------- $name: ok"; else echo "---------- $name: FAILED"; FAILED+=("$name"); fi
}

for s in $STAGES; do
  case "$s" in
    host)  stage host python3 -I tests/test_compare_dictation.py ;;
    build) stage build docker compose build stt ;;
    cpu)   stage cpu docker compose --profile test run --rm test-cpu ;;
    npu)
      TOUCHED_STT=1
      docker compose stop stt >/dev/null 2>&1
      stage npu docker compose --profile test run --rm test-npu ;;
    serve)
      TOUCHED_STT=1
      clips=(); [ -n "${CLIPS:-}" ] && clips=(--clips "$CLIPS")
      serve_torch() {
        NPU_STT_DECODER=torch docker compose up -d stt >/dev/null 2>&1 && wait_ready &&
          python3 -I tests/test_serve_http.py --url "$URL" --expect-decoder torch \
            "${clips[@]}" --save "$TMP/torch.json"
      }
      serve_ct2() {
        NPU_STT_DECODER=ct2 docker compose up -d stt >/dev/null 2>&1 && wait_ready &&
          python3 -I tests/test_serve_http.py --url "$URL" --expect-decoder ct2 \
            "${clips[@]}" --against "$TMP/torch.json"
      }
      stage serve-torch serve_torch
      stage serve-ct2 serve_ct2 ;;
    *) echo "unknown stage: $s (host build cpu npu serve)" >&2; exit 2 ;;
  esac
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL STAGES PASSED: $STAGES"; exit 0; fi
echo "FAILED STAGES: ${FAILED[*]}"; exit 1
