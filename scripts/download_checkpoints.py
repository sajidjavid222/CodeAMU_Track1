#!/usr/bin/env python3
"""Download the 24 Track 1 detector checkpoints (≈34 GB) from the Hugging Face Hub into checkpoints/detectors/.
    python scripts/download_checkpoints.py [--repo sajidjavid/CodeAMU-IndoML2026-Track1]
The Track 2 enhancer used by the noise-meter step (checkpoints/enhancer/, 47 MB) ships with this repository."""
import argparse
import os

from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="sajidjavid/CodeAMU-IndoML2026-Track1")
ap.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"))
a = ap.parse_args()
snapshot_download(a.repo, repo_type="model", local_dir=a.dest, allow_patterns=["detectors/*"])
print("checkpoints ->", a.dest)
