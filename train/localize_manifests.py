#!/usr/bin/env python3
"""Install the training manifests (train/manifests/*.jsonl.gz, exactly as used) under work/ with repo-relative paths.

Paths in the released manifests are either relative to work/train ("audio/<clip>.wav", the --root work/train
manifests) or were absolute on our cluster; the latter are rewritten to repo-relative paths:
  work/train/audio/<clip>.wav       Vaani-Noise-Event-Dataset clips   (train/prepare_data.py)
  work/synth_loud/audio/<clip>.wav  synthetic augmentation            (train/make_synthetic.py)
  data/validation/...               official validation release       (download from the competition page)
    python train/localize_manifests.py
"""
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD = "/home/sajidj/ICASSP/New_Folder/IndoML2026_Track1/"

for f in sorted(os.listdir(f"{HERE}/manifests")):
    rel = "work/" + f[:-3].replace("__", "/")          # e.g. train__train_split.jsonl -> work/train/train_split.jsonl
    dst = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.open(f"{HERE}/manifests/{f}", "rt") as fi, open(dst, "w") as fo:
        n = 0
        for line in fi:
            fo.write(line.replace(OLD, "")); n += 1
    print(f"{rel}: {n} records")
