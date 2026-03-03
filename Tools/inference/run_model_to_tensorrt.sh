#!/usr/bin/env bash
set -euo pipefail


PYTHON_BIN="${PYTHON_BIN:-python3}"

# Update these as needed.
CHECKPOINT_PATH="/workspace/Benchmarks/ActionRecognition/MTRSAP/checkpoints/resnet_segmentation/resnet_30s_segmentation_mtrsap.pt"
OUTPUT_PATH="/workspace/Benchmarks/ActionRecognition/MTRSAP/checkpoints/resnet_segmentation/resnet_30s_segmentation_mtrsap_trt.ts"

# Optional conversion settings.
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-30}"
NHEAD="${NHEAD:-4}"
DROPOUT="${DROPOUT:-0.1}"
FP16="${FP16:-1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: '${PYTHON_BIN}' is not available in PATH." >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Error: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

extra_args=()
if [[ "${FP16}" == "1" ]]; then
  extra_args+=(--fp16)
fi

exec "${PYTHON_BIN}" -u "./model_to_tensorRT.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --nhead "${NHEAD}" \
  --dropout "${DROPOUT}" \
  "${extra_args[@]}" \
  "$@"
