#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

APP_HOME="${APP_HOME:-/workspace/EgoEMS}"
BACKEND="${BACKEND:-detr-trt}"
ENGINE_PATH="${ENGINE_PATH:-${APP_HOME}/Tools/inference/checkpoints/ems_finetuned_detr_trt.ts}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
WARMUP="${WARMUP:-3}"
DETR_VERSION="${DETR_VERSION:-ems}"
DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.7}"
ENGINE_HEIGHT="${ENGINE_HEIGHT:-480}"
ENGINE_WIDTH="${ENGINE_WIDTH:-640}"

if [[ ! -d "${APP_HOME}" ]]; then
  echo "Error: APP_HOME does not exist: ${APP_HOME}" >&2
  echo "Mount the repo into the container, for example:" >&2
  echo "  -v /mnt/f/repos/EgoEMS:${APP_HOME}" >&2
  exit 1
fi

if [[ ! -f "${APP_HOME}/Tools/inference/servers/container_inference_server.py" ]]; then
  echo "Error: server entrypoint not found under ${APP_HOME}" >&2
  echo "Expected: ${APP_HOME}/Tools/inference/servers/container_inference_server.py" >&2
  exit 1
fi

if [[ ! -f "${ENGINE_PATH}" ]]; then
  echo "Error: TensorRT engine not found: ${ENGINE_PATH}" >&2
  exit 1
fi

cd "${APP_HOME}"

echo "[server] starting container inference server"
echo "[server] app home   : ${APP_HOME}"
echo "[server] backend    : ${BACKEND}"
echo "[server] engine     : ${ENGINE_PATH}"
echo "[server] host:port  : ${SERVER_HOST}:${SERVER_PORT}"
echo "[server] device     : ${DEVICE}"
echo "[server] detr ver   : ${DETR_VERSION}"
echo "[server] threshold  : ${DETECTION_THRESHOLD}"
echo "[server] input size : ${ENGINE_WIDTH}x${ENGINE_HEIGHT}"

exec python Tools/inference/servers/container_inference_server.py \
  --backend "${BACKEND}" \
  --engine "${ENGINE_PATH}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --device "${DEVICE}" \
  --warmup "${WARMUP}" \
  --detr-version "${DETR_VERSION}" \
  --threshold "${DETECTION_THRESHOLD}" \
  --engine-height "${ENGINE_HEIGHT}" \
  --engine-width "${ENGINE_WIDTH}"
