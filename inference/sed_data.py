"""Dataset / collate for the multi-tier Vaani noise-event manifest.

Each item yields a waveform plus three kinds of supervision, and a `tier`
telling the training loop which of them are trustworthy:

  gold   -- frame targets from verified timestamps  (strong loss, full weight)
  silver -- frame targets from unverified timestamps(strong loss, down-weighted)
  bronze -- no timestamps at all                    (weak/clip loss only)

`has_strong` is the flag the loss reads; bronze rows still carry a valid weak
target, which is the whole point of the tier.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from sed_model import HOP, SR, TIME_POOL

# Category-head order, identical to train/prepare_data.py CATEGORIES (prepare_data needs pyarrow, not an inference dep).
CATEGORIES = ["animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech"]

# SpecAugment-style masking, off by default so every existing run reproduces.
SPECAUG = os.environ.get("SED_SPECAUG", "0") == "1"

FRAME_SEC = HOP * TIME_POOL / SR          # 0.02 s
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


def load_manifest(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def n_frames_for(n_samples):
    return max(1, (n_samples // HOP + 1) // TIME_POOL)


def events_to_boundaries(events, n_frames, frame_sec=FRAME_SEC, offset=0.0,
                         sigma_frames=2.0):
    """Gaussian onset/offset targets (soft boundary supervision).

    A hard one-frame spike is nearly impossible to fit and gives almost no
    gradient; a Gaussian of sigma ~2 frames (40 ms) is the standard onset
    detection target and matches the ~89 ms median tolerance the metric allows.
    """
    on = np.zeros(n_frames, dtype=np.float32)
    off = np.zeros(n_frames, dtype=np.float32)
    rad = int(np.ceil(3 * sigma_frames))
    for e in events:
        for t, arr in ((e["onset"], on), (e["offset"], off)):
            c = (t - offset) / frame_sec
            i0, i1 = max(0, int(np.floor(c)) - rad), min(n_frames, int(np.ceil(c)) + rad + 1)
            if i1 <= i0:
                continue
            idx = np.arange(i0, i1)
            arr[i0:i1] = np.maximum(arr[i0:i1],
                                    np.exp(-0.5 * ((idx - c) / sigma_frames) ** 2))
    return on, off


def events_to_frames(events, n_frames, frame_sec=FRAME_SEC, offset=0.0):
    """Binary + per-category frame targets for a list of event dicts."""
    y = np.zeros(n_frames, dtype=np.float32)
    yc = np.zeros((n_frames, len(CATEGORIES)), dtype=np.float32)
    for e in events:
        i0 = int(np.floor((e["onset"] - offset) / frame_sec))
        i1 = int(np.ceil((e["offset"] - offset) / frame_sec))
        i0, i1 = max(0, i0), min(n_frames, i1)
        if i1 <= i0:
            continue
        y[i0:i1] = 1.0
        ci = CAT_INDEX.get(e.get("category"))
        if ci is not None:
            yc[i0:i1, ci] = 1.0
    return y, yc


class VaaniSED(Dataset):
    def __init__(self, records, root, max_dur=10.0, train=True,
                 augment=True, sr=SR, silver_edge_sec=0.0):
        self.records = records
        self.root = root
        self.max_len = int(max_dur * sr)
        self.train = train
        self.augment = augment and train
        self.sr = sr
        self.silver_edge_frames = int(round(silver_edge_sec / FRAME_SEC))

    def __len__(self):
        return len(self.records)

    def _read(self, rec):
        path = rec["path"]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr

    def __getitem__(self, i):
        rec = self.records[i]
        y, sr = self._read(rec)
        events = rec.get("events", [])
        offset_sec = 0.0

        # random crop for long clips (keeps memory bounded, acts as augmentation)
        if len(y) > self.max_len:
            start = random.randint(0, len(y) - self.max_len) if self.train else 0
            y = y[start:start + self.max_len]
            offset_sec = start / sr
            lo, hi = offset_sec, offset_sec + self.max_len / sr
            events = [e for e in events if e["offset"] > lo and e["onset"] < hi]

        if self.augment:
            y = y * (10 ** (random.uniform(-6, 6) / 20.0))     # +/-6 dB gain
            if random.random() < 0.3:                           # mild white noise
                y = y + np.random.randn(len(y)).astype(np.float32) * \
                    (10 ** (random.uniform(-45, -30) / 20.0))
            if SPECAUG:
                # SpecAugment-style masking, applied on the WAVEFORM so it works
                # for every backbone (Whisper/WavLM/BEATs each build their own
                # features internally, so there is no shared spectrogram to mask).
                # Every model here overfits by epoch 5-6 with only gain + faint
                # white noise as augmentation; this is the standard remedy and
                # had never been tried on this pipeline.
                #
                # Labels are deliberately NOT changed: the event is still there,
                # the network just has to infer it from partial evidence. That is
                # what makes masking a regulariser rather than label noise.
                n = len(y)
                for _ in range(random.randint(0, 2)):           # time masks
                    w = int(n * random.uniform(0.02, 0.08))
                    if w > 1 and n > w:
                        i = random.randint(0, n - w)
                        y[i:i + w] = 0.0
                if random.random() < 0.5:                       # band-stop ~ freq mask
                    Y = np.fft.rfft(y)
                    nb = len(Y)
                    for _ in range(random.randint(1, 2)):
                        bw = int(nb * random.uniform(0.03, 0.12))
                        if bw > 1 and nb > bw:
                            f0 = random.randint(0, nb - bw)
                            Y[f0:f0 + bw] = 0.0
                    y = np.fft.irfft(Y, n=n).astype(np.float32)
            peak = np.abs(y).max()
            if peak > 1.0:
                y = y / peak

        nf = n_frames_for(len(y))
        tier = rec.get("tier", "silver")
        has_strong = tier in ("gold", "silver")
        fw = np.ones(nf, dtype=np.float32)      # per-frame loss weight

        if has_strong:
            fr, frc = events_to_frames(events, nf, offset=offset_sec)
            on_t, off_t = events_to_boundaries(events, nf, offset=offset_sec)
            # Silver timestamps are explicitly UNVERIFIED, but only their edges
            # are suspect -- the interior of a marked event really is noise and
            # the far exterior really is not. Down-weighting the whole clip (as
            # before) discards that reliable signal while still absorbing the
            # bad edges. Instead, drop only the frames near a silver boundary.
            if tier == "silver" and self.silver_edge_frames > 0:
                w = int(self.silver_edge_frames)
                for e in events:
                    for t in (e["onset"], e["offset"]):
                        c = int(round((t - offset_sec) / FRAME_SEC))
                        fw[max(0, c - w):min(nf, c + w + 1)] = 0.0
        else:
            fr = np.zeros(nf, dtype=np.float32)
            frc = np.zeros((nf, len(CATEGORIES)), dtype=np.float32)
            on_t = np.zeros(nf, dtype=np.float32)
            off_t = np.zeros(nf, dtype=np.float32)

        # weak (clip-level) targets exist for every tier
        cats = rec.get("categories") or [e.get("category") for e in events]
        weak_cat = np.zeros(len(CATEGORIES), dtype=np.float32)
        for c in cats:
            ci = CAT_INDEX.get(c)
            if ci is not None:
                weak_cat[ci] = 1.0
        weak = 1.0 if (events or cats or rec.get("weak_tags")) else 0.0

        return {
            "wav": torch.from_numpy(np.ascontiguousarray(y)),
            "frame": torch.from_numpy(fr),
            "frame_cat": torch.from_numpy(frc),
            "frame_w": torch.from_numpy(fw),
            "onset_t": torch.from_numpy(on_t),
            "offset_t": torch.from_numpy(off_t),
            "weak": torch.tensor(weak, dtype=torch.float32),
            "weak_cat": torch.from_numpy(weak_cat),
            "has_strong": torch.tensor(float(has_strong)),
            "tier": tier,
            "clip_id": rec["clip_id"],
            "duration": float(rec.get("duration") or len(y) / sr),
        }


def collate(batch):
    """Pad to the longest waveform in the batch; carry a frame-level mask."""
    maxs = max(b["wav"].shape[0] for b in batch)
    maxf = max(b["frame"].shape[0] for b in batch)
    B, C = len(batch), len(CATEGORIES)

    wav = torch.zeros(B, maxs)
    frame = torch.zeros(B, maxf)
    frame_w = torch.ones(B, maxf)
    onset_t = torch.zeros(B, maxf)
    offset_t = torch.zeros(B, maxf)
    frame_cat = torch.zeros(B, maxf, C)
    mask = torch.zeros(B, maxf, dtype=torch.bool)
    weak = torch.zeros(B)
    weak_cat = torch.zeros(B, C)
    has_strong = torch.zeros(B)

    for i, b in enumerate(batch):
        n, f = b["wav"].shape[0], b["frame"].shape[0]
        wav[i, :n] = b["wav"]
        frame[i, :f] = b["frame"]
        frame_w[i, :f] = b["frame_w"]
        onset_t[i, :f] = b["onset_t"]
        offset_t[i, :f] = b["offset_t"]
        frame_cat[i, :f] = b["frame_cat"]
        mask[i, :f] = True
        weak[i] = b["weak"]
        weak_cat[i] = b["weak_cat"]
        has_strong[i] = b["has_strong"]

    return {"wav": wav, "frame": frame, "frame_cat": frame_cat, "mask": mask,
            "onset_t": onset_t, "offset_t": offset_t, "frame_w": frame_w,
            "weak": weak, "weak_cat": weak_cat, "has_strong": has_strong,
            "tier": [b["tier"] for b in batch],
            "clip_id": [b["clip_id"] for b in batch],
            "duration": [b["duration"] for b in batch]}


def split_manifest(records, val_frac=0.1, seed=0, prefer_gold_val=True):
    """Hold out a validation set.

    Validation is drawn from gold (falling back to silver) so the tuned
    decoding thresholds are fitted against trustworthy timestamps -- tuning on
    silver would bake annotation noise into the submission.
    """
    rng = random.Random(seed)
    gold = [r for r in records if r.get("tier") == "gold"]
    pool = gold if (prefer_gold_val and len(gold) >= 200) else \
        [r for r in records if r.get("tier") in ("gold", "silver")]

    ids = sorted({r["clip_id"] for r in pool})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])

    val = [r for r in records if r["clip_id"] in val_ids]
    train = [r for r in records if r["clip_id"] not in val_ids]
    return train, val
