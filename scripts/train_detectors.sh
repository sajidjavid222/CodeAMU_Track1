#!/usr/bin/env bash
# CodeAMU - Track 1: train every sound-event detector of the final ensemble (24 models) and the 5 ancestors
# they were fine-tuned from, in dependency order. Configs: train/configs/detectors.yaml (exact arguments stored
# inside each released checkpoint). Each command ran as one job on 1x NVIDIA H200 (MIG 3g.71gb slice or full GPU).
# Prerequisite: bash scripts/prepare_training_data.sh (Vaani WAVs + manifests + synthetic augmentation).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# rb_beats  (ancestor, not in the final ensemble)
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split4k.jsonl \
    --root work/train \
    --out runs/rb_beats \
    --val-manifest work/train/val_split4k.jsonl \
    --model beats_hybrid \
    --ssl-name openai/whisper-medium \
    --lr-encoder 1e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 8 \
    --batch-size 8 \
    --lr 0.0007 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --beats-ckpt "$BEATS_CKPT" \
    --amp \
    --mean-teacher \
    --resume

# rb_beats_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split4k.jsonl \
    --root work/train \
    --out runs/rb_beats_gold \
    --val-manifest work/train/val_split4k.jsonl \
    --model beats_hybrid \
    --ssl-name openai/whisper-medium \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 6 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/rb_beats/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --beats-ckpt "$BEATS_CKPT" \
    --amp \
    --gold-only \
    --resume

# beats_full_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/all_gold_13k.jsonl \
    --root . \
    --out runs/beats_full_gold \
    --val-manifest work/official_val.jsonl \
    --model beats_hybrid \
    --ssl-name openai/whisper-medium \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 5 \
    --batch-size 6 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/rb_beats_gold/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --beats-ckpt "$BEATS_CKPT" \
    --amp \
    --gold-only \
    --resume

# bg_lg3_syn
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full_plus_loud.jsonl \
    --root . \
    --out runs/bg_lg3_syn \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 8e-06 \
    --freeze-encoder-epochs 1 \
    --epochs 6 \
    --batch-size 8 \
    --lr 0.0005 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# bg_lg3_syn_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full.jsonl \
    --root . \
    --out runs/bg_lg3_syn_gold \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 6 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/bg_lg3_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# bg_sml_syn
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full_plus_loud.jsonl \
    --root . \
    --out runs/bg_sml_syn \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-small \
    --lr-encoder 2e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 8 \
    --batch-size 24 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# cb_big
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full_plus_loud.jsonl \
    --root . \
    --out runs/cb_big \
    --val-manifest work/official_val.jsonl \
    --model crnn_big \
    --ssl-name none \
    --lr-encoder 0.001 \
    --freeze-encoder-epochs 1 \
    --epochs 34 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# cb_s7
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full_plus_loud.jsonl \
    --root . \
    --out runs/cb_s7 \
    --val-manifest work/official_val.jsonl \
    --model crnn_big \
    --ssl-name none \
    --lr-encoder 0.001 \
    --freeze-encoder-epochs 1 \
    --epochs 34 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 777 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# cb_s7_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full.jsonl \
    --root . \
    --out runs/cb_s7_gold \
    --val-manifest work/official_val.jsonl \
    --model crnn_big \
    --ssl-name none \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 48 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 777 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/cb_s7/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# gv_lg3_A
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/gold_plus_valA.jsonl \
    --root . \
    --out runs/gv_lg3_A \
    --val-manifest work/val_holdA.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 4 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/bg_lg3_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# gv_lg3_B
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/gold_plus_valB.jsonl \
    --root . \
    --out runs/gv_lg3_B \
    --val-manifest work/val_holdB.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 4 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/bg_lg3_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# ov_med_syn
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train_full_plus_loud.jsonl \
    --root . \
    --out runs/ov_med_syn \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-medium \
    --lr-encoder 1e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 8 \
    --batch-size 16 \
    --lr 0.0007 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# gv_med_A
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/gold_plus_valA.jsonl \
    --root . \
    --out runs/gv_med_A \
    --val-manifest work/val_holdA.jsonl \
    --model whisper \
    --ssl-name openai/whisper-medium \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 6 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/ov_med_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# gv_med_B
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/gold_plus_valB.jsonl \
    --root . \
    --out runs/gv_med_B \
    --val-manifest work/val_holdB.jsonl \
    --model whisper \
    --ssl-name openai/whisper-medium \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 6 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/ov_med_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# gv_sml_A
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/gold_plus_valA.jsonl \
    --root . \
    --out runs/gv_sml_A \
    --val-manifest work/val_holdA.jsonl \
    --model whisper \
    --ssl-name openai/whisper-small \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 8 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/bg_sml_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# gv_sml_B
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/gold_plus_valB.jsonl \
    --root . \
    --out runs/gv_sml_B \
    --val-manifest work/val_holdB.jsonl \
    --model whisper \
    --ssl-name openai/whisper-small \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 8 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/bg_sml_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# hr_med_syn
SED_TIME_POOL=1 \
    python train/train.py \
    --manifest work/train_full_plus_loud.jsonl \
    --root . \
    --out runs/hr_med_syn \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-medium \
    --lr-encoder 1e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 6 \
    --batch-size 8 \
    --lr 0.0007 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# hr_med_syn_gold
SED_TIME_POOL=1 \
    python train/train.py \
    --manifest work/train_full.jsonl \
    --root . \
    --out runs/hr_med_syn_gold \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-medium \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 4 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 5 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/hr_med_syn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# v9_lg2  (ancestor, not in the final ensemble)
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v9_lg2 \
    --val-manifest work/train/val_split.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v2 \
    --lr-encoder 8e-06 \
    --freeze-encoder-epochs 1 \
    --epochs 10 \
    --batch-size 16 \
    --lr 0.0005 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 5 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# v9_lg2_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v9_lg2_gold \
    --val-manifest work/train/val_split.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v2 \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 4 \
    --batch-size 8 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 5 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/v9_lg2/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# lg2_full_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/all_gold_13k.jsonl \
    --root . \
    --out runs/lg2_full_gold \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v2 \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 6 \
    --batch-size 8 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/v9_lg2_gold/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# v9_lg3_s21
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v9_lg3_s21 \
    --val-manifest work/train/val_split.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 8e-06 \
    --freeze-encoder-epochs 1 \
    --epochs 12 \
    --batch-size 16 \
    --lr 0.0005 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 21 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# lg3_full_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/all_gold_13k.jsonl \
    --root . \
    --out runs/lg3_full_gold \
    --val-manifest work/official_val.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 5e-06 \
    --freeze-encoder-epochs 0 \
    --epochs 6 \
    --batch-size 8 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/v9_lg3_s21/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

# v2_crnn  (ancestor, not in the final ensemble)
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v2_crnn \
    --val-manifest work/train/val_split.jsonl \
    --model crnn \
    --ssl-name microsoft/wavlm-base-plus \
    --lr-encoder 3e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 40 \
    --batch-size 96 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --amp \
    --resume

# v2_crnn_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v2_crnn_gold \
    --val-manifest work/train/val_split.jsonl \
    --model crnn \
    --ssl-name microsoft/wavlm-base-plus \
    --lr-encoder 3e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 12 \
    --batch-size 64 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --init-from runs/v2_crnn/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --amp \
    --gold-only \
    --resume

# v4_wavlm  (ancestor, not in the final ensemble)
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v4_wavlm \
    --val-manifest work/train/val_split.jsonl \
    --model ssl \
    --ssl-name microsoft/wavlm-base-plus \
    --lr-encoder 3e-05 \
    --freeze-encoder-epochs 1 \
    --epochs 10 \
    --batch-size 48 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 4 \
    --amp \
    --resume

# v4_wavlm_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v4_wavlm_gold \
    --val-manifest work/train/val_split.jsonl \
    --model ssl \
    --ssl-name microsoft/wavlm-base-plus \
    --lr-encoder 1e-05 \
    --freeze-encoder-epochs 0 \
    --epochs 6 \
    --batch-size 32 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 10 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --init-from runs/v4_wavlm/swa.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 2 \
    --amp \
    --gold-only \
    --resume

# v8_whis_lg  (ancestor, not in the final ensemble)
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v8_whis_lg \
    --val-manifest work/train/val_split.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 6e-06 \
    --freeze-encoder-epochs 1 \
    --epochs 4 \
    --batch-size 6 \
    --lr 0.0005 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.25 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --resume

# v8_whis_lg_gold
SED_TIME_POOL=2 \
    python train/train.py \
    --manifest work/train/train_split.jsonl \
    --root work/train \
    --out runs/v8_whis_lg_gold \
    --val-manifest work/train/val_split.jsonl \
    --model whisper \
    --ssl-name openai/whisper-large-v3 \
    --lr-encoder 1e-05 \
    --freeze-encoder-epochs 0 \
    --epochs 2 \
    --batch-size 4 \
    --lr 0.0002 \
    --weight-decay 0.01 \
    --workers 6 \
    --max-dur 10.0 \
    --val-frac 0.1 \
    --seed 0 \
    --w-gold 1.0 \
    --w-silver 0.0 \
    --w-cat 0.3 \
    --w-weak 0.5 \
    --w-bound 1.0 \
    --silver-edge-sec 0.0 \
    --init-from runs/v8_whis_lg/best.pt \
    --w-cons 1.0 \
    --ema-decay 0.999 \
    --swa-start 0 \
    --amp \
    --gold-only \
    --resume

