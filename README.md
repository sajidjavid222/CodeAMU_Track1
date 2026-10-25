# CodeAMU — IndoML 2026 Datathon, Track 1: Noise Event Detection

**Team CodeAMU** — Shah Ahmed Shakir Abu Asim Khan (Aligarh Muslim University), Sajid Javid (IIIT Delhi),
Osama Zaheer (Binary Semantics), Ahraz Shamim (Vecmocon Technologies)
**Primary contact:** Sajid Javid — sajidj@iiitd.ac.in

**Final standing: 1st place, Phase 2 (final) leaderboard — Combined 1.6173** (Event F1 0.7895 + Segment Dice 0.8278;
natural clips F1 0.5197 / Dice 0.7256, synthetic clips F1 0.9981 / Dice 0.9966), submission `v93_banksyn7`.
Technical report: [`docs/CodeAMU_track_1_1.pdf`](docs/CodeAMU_track_1_1.pdf).
Detector checkpoints: <https://huggingface.co/sajidjavid/CodeAMU-IndoML2026-Track1>. Companion Track 2 system: <https://github.com/sajidjavid222/CodeAMU_Track2>.

## System overview

```
test clip ─► 24 sound-event detectors (Whisper / WavLM / BEATs / CRNN encoders; presence + onset + offset heads)
          ─► router (random forest: synthetic vs natural)
               ├─ natural branch : weighted 16-model + 3 full-gold + CRNN blend → hysteresis decoding → onset/offset
               │                   splitting, duration snapping, interior carving, boundary regressor → post-processing
               │                   (duplicate NMS, confidence false-alarm filter, offset cut, collar alignment)
               └─ synthetic branch: 10-model mean → decoding → splitting / snapping
          ─► noise-bank verification: FFT matching against the official validation noise pool; spans of
               energy-verified matches replace / complete events (synthetic and natural clips)
          ─► noise meter: Track 2 enhancer drops bank-unsupported synthetic events containing no noise
          ─► empty natural clips: one event over the ensemble's activity envelope
          ─► predictions.jsonl
```

The full description (losses, weights, thresholds, calibration) is in the report and `configs/model_config.yaml`.

## Repository layout

```
CodeAMU_Track1/
├── README.md
├── requirements_train.txt / requirements_inference.txt
├── train/
│   ├── train.py, sed_model.py, sed_data.py     # detector training (multi-tier loss), architectures, data loading
│   ├── beats/                                  # BEATs encoder code (Microsoft UniLM, MIT licence)
│   ├── prepare_data.py                         # Vaani-Noise-Event parquet -> 16 kHz WAVs + manifest
│   ├── make_synthetic.py                       # synthetic augmentation (exact event boundaries)
│   ├── localize_manifests.py, manifests/       # the exact training / validation manifests (gzip)
│   ├── synth_clf.py                            # router features + training
│   ├── train_detector_router.py, detector_router.py   # second router (detector outputs), Track 2 Phase 2 routing
│   ├── train_boundary_regressor.py             # boundary regressor
│   └── configs/detectors.yaml                  # exact training arguments of all 29 runs (24 final + 5 ancestors)
├── inference/
│   ├── run_pipeline.py                         # entry point (stages 1-5, resumable)
│   ├── infer.py                                # stage 1: detector frame probabilities
│   ├── t1_v67_decode.py                        # stage 2: routing, ensembles, decoding
│   ├── bank_match.py                           # stage 3: noise-bank matching
│   ├── t2_bank_subtract.py                     # stage 4: GLS bank subtraction spans
│   ├── build_v68.py ... build_v93_banksyn7.py  # stage 5: refinement chain
│   ├── t1_bank_extent_probe.py, t1_syn_fp_probe.py, enhance.py   # energy test, noise meter (Track 2 enhancer)
│   ├── route_combined.py, detector_router.py   # combined routing for the Track 2 inputs (--track2_inputs)
│   └── (helpers) postprocess.py, dur_snap.py, boundary_regressor.py, bank_merge.py, metrics.py, synth_clf.py, ...
├── assets/                                     # synth_clf.pkl (router), detector_router.pkl, boundary_regressor.pkl, route_v43.json, syn_branch_v57.json
├── checkpoints/
│   ├── detectors/<name>/best.pt                # 24 detectors, ≈34 GB -> python scripts/download_checkpoints.py
│   └── enhancer/t2_model_{0,1,2}.pt            # Track 2 TF-DPCRN ensemble used as the noise meter (shipped)
├── configs/model_config.yaml
├── scripts/run_inference.sh, download_checkpoints.py, prepare_training_data.sh, train_detectors.sh
└── docs/CodeAMU_track_1_1.pdf
```

## Environment and hardware

| | |
|---|---|
| Python | 3.10.12 |
| PyTorch / CUDA / cuDNN | 2.11.0+cu128 / CUDA 12.8 / cuDNN 9.19 |
| Training GPUs | NVIDIA H200 (141 GB). Base detectors: one full H200, 16 CPU cores, 250 GB RAM, ≤ 24 h jobs. Fine-tuning: H200 MIG `3g.71gb` slices (71 GB), 10 CPU cores, 60 GB RAM, ≤ 6 h jobs. SLURM cluster. |
| Inference | stage 1 (24 detectors) on one H200 MIG `3g.71gb` slice: ≈2.5 h for the 10.8 h test set when run one after another as `run_pipeline.py` does (≈50 min with three detectors in parallel); stages 2-5 on CPU, dominated by noise-bank matching (several hours on CPU) |
| OS | Ubuntu 22.04.4 LTS, x86-64 |
| System packages | `libsndfile1` |

**Pretrained models**

| Model | Hugging Face source (revision) | Used by |
|---|---|---|
| Whisper small | `openai/whisper-small` (`973afd24965f72e36ca33b3055d56a652f456b4d`) | encoder of bg_sml_syn, gv_sml_A/B |
| Whisper medium | `openai/whisper-medium` (`abdf7c39ab9d0397620ccaea8974cc764cd0953e`) | ov_med_syn, gv_med_A/B, hr_med_syn(_gold), BEATs hybrids |
| Whisper large-v2 | `openai/whisper-large-v2` (`ae4642769ce2ad8fc292556ccea8e901f1530655`) | v9_lg2_gold, lg2_full_gold |
| Whisper large-v3 | `openai/whisper-large-v3` (`06f233fe06e710322aca913c1bc4249a0d71fce1`) | v8/v9 large, gv_lg3_A/B, lg3_full_gold, bg_lg3_syn(_gold) |
| WavLM base-plus | `microsoft/wavlm-base-plus` (`4c66d4806a428f2e922ccfa1a962776e232d487b`) | v4_wavlm_gold |
| BEATs iter3+ (AS2M fine-tuned, cpt2) | `WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2` (`172e4071a2912392d9151c70750b9a12f55fc54e`), mirror of Microsoft BEATs | rb_beats_gold, beats_full_gold |
| Track 2 TF-DPCRN enhancer | our model, `checkpoints/enhancer/` (CodeAMU Track 2) | noise meter |

Encoders are downloaded automatically by `transformers` / `huggingface_hub` on first use.

**Data**: `ARTPARK-IISc/Vaani-Noise-Event-Dataset` (90,635 training clips; gold 11,111 / silver 61,640 / bronze 17,884)
and the official validation release (2,501 clips: 1,806 natural, 695 synthetic). The validation release is also
used to (i) train cross-fitted fold models and the full-gold models, (ii) train the router and boundary regressor,
(iii) build the noise bank. No other external data.

## Setup

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchaudio==2.11.0
pip install -r requirements_inference.txt            # or requirements_train.txt
python scripts/download_checkpoints.py               # 24 detectors (≈34 GB) -> checkpoints/detectors/
```

## Inference

```bash
bash scripts/run_inference.sh /path/to/test/audio /path/to/run_dir /path/to/validation
# = python inference/run_pipeline.py --audio_dir ... --run_dir ... --val_dir ... [--stages 1,2,3,4,5]
```

**Inputs**
* test clips: one 16 kHz mono WAV per clip (`<clip_id>.wav`) in `--audio_dir`;
* `--val_dir`: the official validation release (`validationMetadata.json`, `syntheticNoiseAudio/`,
  `syntheticCleanRefAudio/`, `naturalNoisyAudio/`) — source of the noise bank.

**Outputs** (`<run_dir>/output/`)
* `predictions.jsonl`: one line per clip, `{"clip_id": ..., "events": [{"onset": s, "offset": s}, ...]}`;
* `submission.zip`: `predictions.jsonl` at the zip root (Codabench format).
Intermediate artefacts (`work/`, `submission/v67 ... v93_banksyn7/`) stay in the run dir; every stage is skipped if its
output exists, so a run can be resumed.

**Track 2 inputs.** The CodeAMU Track 2 system takes this system's event timestamps and synthetic/natural routing as
input. For the Track 2 Phase 2 audio they were made with

```bash
python inference/run_pipeline.py --audio_dir <track2 phase2 audio> --run_dir <dir> --val_dir <validation> --stages 1,2 --track2_inputs
```

which writes `<dir>/output/track2_inputs/{track1_predictions.jsonl, route.json}`: the audio router's probability is
averaged with a second router that reads detector outputs (`assets/detector_router.pkl`), clips at >= 0.61 are
synthetic, and stage 2 is re-run under that routing. On the cached Phase 2 detector outputs this reproduces the files
shipped in [`CodeAMU_Track2/inputs/phase2/`](https://github.com/sajidjavid222/CodeAMU_Track2/tree/main/inputs/phase2) byte for byte (3,178 clips, 1,635 synthetic, 13,008 events).

**Reproduction check** (Track 1 test audio, this repository's code):
* Stage 1: four of the 24 detectors, one per architecture type (`lg3_full_gold`, `hr_med_syn`, `rb_beats_gold`,
  `cb_big`), re-run with `inference/infer.py` on one H200 `3g.71gb` slice, reproduce the cached frame probabilities of
  all 5,517 clips bit for bit.
* Stages 2 and 5: from the cached detector outputs and noise-bank matches, `run_pipeline.py` regenerates every
  intermediate submission (v67, v69, v70, v71, v74, v76, v80, v93, v93_banksyn2 ... v93_banksyn7) identically on all
  5,517 clips; the final output equals the submitted `v93_banksyn7` (13,449 events).

## Training

```bash
bash scripts/prepare_training_data.sh     # Vaani WAVs + manifests + synthetic augmentation (11,762 clips, 21.4 h)
bash scripts/train_detectors.sh           # 29 runs in dependency order (exact arguments: train/configs/detectors.yaml)
```

* **Detectors** — `train/train.py`, multi-tier loss: frame BCE on gold (weight 1.0) and silver (0.25) timestamps,
  frame category BCE (0.3), clip-level weak BCE (0.5, the only signal of bronze clips), onset/offset Gaussian-bump BCE
  on gold rows (1.0); AdamW, warm-up with a frozen encoder, bf16; `SED_TIME_POOL=1` for the 10 ms `hr_*` models.
  Fine-tuning chains: synthetic-augmented pre-training → gold-only fine-tuning; fold models `*_A` / `*_B` add one half of
  the official validation set, so the other half stays an honest held-out set; `*_full_gold` use all gold + validation.
* **Router** — `train/synth_clf.py`: random forest on background statistics (noise-floor flatness, high-band energy of
  quiet frames, floor variability, splice jumps), labels = `syntheticData` of the validation release.
  `train/train_detector_router.py` trains the second router (Track 2 inputs only) on cb_big / gv_lg3_A / rb_beats_gold
  outputs for the validation clips (5-fold CV accuracy 99.36 %; the shipped pickle reproduces the original exactly).
* **Boundary regressor** — `train/train_boundary_regressor.py`: gradient boosting (Huber) on matched natural candidate
  events of the validation folds, predicting offset corrections from local probability-track features.
* **Post-processing parameters** — chosen on honest out-of-fold validation (folds A / B, 1,806 natural clips); every
  change was kept only if both folds improved.
