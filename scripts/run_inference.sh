#!/usr/bin/env bash
# CodeAMU - Track 1 inference (reproduces submission v93_banksyn7, Combined 1.6173).
#   bash scripts/run_inference.sh <test_audio_dir> <run_dir> <official_validation_dir> [stages, default 1,2,3,4,5]
# Requires checkpoints/detectors/ (python scripts/download_checkpoints.py) and checkpoints/enhancer/ (shipped).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUDIO="${1:?test audio dir}"; RUN="${2:?run dir}"; VAL="${3:?official validation dir}"; STAGES="${4:-1,2,3,4,5}"
python "$ROOT/inference/run_pipeline.py" --audio_dir "$AUDIO" --run_dir "$RUN" --val_dir "$VAL" \
    --checkpoints "$ROOT/checkpoints" --stages "$STAGES"
echo "predictions: $RUN/output/predictions.jsonl   submission: $RUN/output/submission.zip"
