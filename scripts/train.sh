#!/usr/bin/env bash
set -euo pipefail

variant="${1:-posa_posd}"
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen2-Audio model path or Hub id}"
: "${MANIFEST:?Set MANIFEST to the training CSV manifest}"

python_bin="${PYTHON_BIN:-python}"
work_dir="${WORK_DIR:-artifacts}"
run_root="${RUN_ROOT:-runs}"
audio_root="${AUDIO_ROOT:-}"

case "${variant}" in
  posa) posd_weight=0 ;;
  posa_posd) posd_weight=1 ;;
  *)
    echo "Usage: $0 {posa|posa_posd}" >&2
    exit 2
    ;;
esac

mkdir -p "${work_dir}" "${run_root}"
paired_jsonl="${work_dir}/train_posa.jsonl"

posa_args=(
  -m emotrack.posa
  --manifest "${MANIFEST}"
  --output "${paired_jsonl}"
  --num-samples "${NUM_SAMPLES:-8400}"
  --turns "${TURNS:-4}"
  --seed "${SEED:-42}"
)
if [[ -n "${audio_root}" ]]; then
  posa_args+=(--audio-root "${audio_root}")
fi
"${python_bin}" "${posa_args[@]}"

"${python_bin}" train_qwen2audio.py \
  --model "${MODEL_PATH}" \
  --train-jsonl "${paired_jsonl}" \
  --output-dir "${run_root}/${variant}" \
  --posd-weight "${posd_weight}" \
  --seed "${SEED:-42}"

