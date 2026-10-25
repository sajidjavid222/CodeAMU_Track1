#!/usr/bin/env bash
# CodeAMU - Track 1: training data.
#   1. Vaani-Noise-Event-Dataset (Hugging Face, gated: accept the terms first) -> work/train/audio + manifest
#   2. official validation release -> data/validation (naturalNoisyAudio/, syntheticNoiseAudio/, syntheticCleanRefAudio/,
#      validationMetadata.json); download it from the competition's "Files" page
#   3. synthetic augmentation (11,762 clips, 21.4 h) -> work/synth_loud
#   4. training manifests -> work/*.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
: "${HF_TOKEN:?HF token with access to ARTPARK-IISc/Vaani-Noise-Event-Dataset}"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("ARTPARK-IISc/Vaani-Noise-Event-Dataset", repo_type="dataset", local_dir="data/vaani_noise_event",
                  allow_patterns=["*.parquet"])
PY
python train/prepare_data.py --parquet 'data/vaani_noise_event/**/*.parquet' --out work/train --jobs 8
python train/localize_manifests.py
[ -d data/validation ] || { echo "place the official validation release in data/validation"; exit 1; }
python train/make_synthetic.py --manifest work/train/train_split4k.jsonl --root work/train --out work/synth_loud --n 12000 --seed 0
