"""Run a trained detector over the evaluation clips -> predictions.jsonl (+ zip).

Submission format (competition "Evaluation" page):
    {"clip_id": "vaani_eval_001", "events": [{"onset": 1.24, "offset": 3.81}]}
one JSON object per line, every evaluation clip present exactly once, empty
list allowed. The file must be named predictions.jsonl inside a ZIP.

Inputs, in order of preference:
  --audio-dir  directory of eval wav/flac/mp3 (clip_id = filename stem)
  --manifest   a manifest.jsonl produced by prepare_data.py
  --clip-list  optional text/JSON list of the authoritative clip_ids; any clip
               missing from the audio is still emitted with an empty event list
               so the submission is never short a line.

Long clips are processed in overlapping windows and the probability tracks are
averaged over the overlap, so the model is never asked for more context than it
saw in training.

Example:
    python src/infer.py --ckpt runs/crnn_v1/best.pt --audio-dir data/eval/audio \
        --out submission/predictions.jsonl --zip submission/submission.zip
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile

import numpy as np
import soundfile as sf
import torch

from metrics import write_jsonl_events
from postprocess import decode_frames, snap_events
from sed_data import FRAME_SEC
from sed_model import HOP, SR, TIME_POOL, build_model

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".WAV", ".FLAC", ".MP3")


def find_audio(audio_dir):
    files = []
    for ext in AUDIO_EXT:
        files.extend(glob.glob(os.path.join(audio_dir, "**", f"*{ext}"),
                               recursive=True))
    return sorted(set(files))


def read_audio(path, sr=SR):
    y, in_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if in_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    return y


@torch.no_grad()
def frame_probs(model, wav, device, win_sec=10.0, hop_sec=8.0, amp=False):
    """Frame probability track for one waveform, windowed with overlap-average.

    Deliberately one clip at a time. Batching would need zero-padding, and
    wavlm-base-plus uses `feat_extract_norm="group"`, so the conv frontend
    normalises over the padded region and changes the features of the *real*
    frames -- measured max prob deviation ~1.0, i.e. entirely different
    predictions. An attention_mask does not fix it (HF documents that base
    wav2vec2/WavLM checkpoints should be batched with equal-length inputs
    instead). Padding-free batching would need exact length bucketing, which
    buys nothing on continuously-varying clip durations.
    """
    n = len(wav)
    win, hop = int(win_sec * SR), int(hop_sec * SR)
    n_frames = max(1, (n // HOP + 1) // TIME_POOL)
    acc = np.zeros(n_frames, dtype=np.float64)
    acc_cat = np.zeros((n_frames, 7), dtype=np.float64)
    acc_on = np.zeros(n_frames, dtype=np.float64)
    acc_off = np.zeros(n_frames, dtype=np.float64)
    cnt = np.zeros(n_frames, dtype=np.float64)

    starts = [0] if n <= win else list(range(0, max(1, n - win + hop), hop))
    for s in starts:
        seg = wav[s:s + win]
        if len(seg) < int(0.1 * SR):                    # ignore slivers
            continue
        x = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0).to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(x)
        p = torch.sigmoid(out["strong_logit"].float()).squeeze(0).cpu().numpy()
        pc = (torch.sigmoid(out["cat_logit"].float()).squeeze(0).cpu().numpy()
              if "cat_logit" in out else None)
        po = torch.sigmoid(out["onset_logit"].float()).squeeze(0).cpu().numpy() \
            if "onset_logit" in out else np.zeros_like(p)
        pf = torch.sigmoid(out["offset_logit"].float()).squeeze(0).cpu().numpy() \
            if "offset_logit" in out else np.zeros_like(p)
        f0 = int(round(s / SR / FRAME_SEC))
        f1 = min(n_frames, f0 + len(p))
        if f1 > f0:
            acc[f0:f1] += p[:f1 - f0]
            if pc is not None:
                acc_cat[f0:f1] += pc[:f1 - f0]
            acc_on[f0:f1] += po[:f1 - f0]
            acc_off[f0:f1] += pf[:f1 - f0]
            cnt[f0:f1] += 1
    cnt[cnt == 0] = 1
    c = cnt[:, None]
    return acc / cnt, acc_on / cnt, acc_off / cnt, acc_cat / c


def load_clip_list(path):
    if not path:
        return None
    with open(path) as fh:
        txt = fh.read().strip()
    if txt.startswith("["):
        return [str(x) for x in json.loads(txt)]
    if txt.startswith("{"):                             # jsonl of records
        ids = []
        for line in txt.splitlines():
            if line.strip():
                r = json.loads(line)
                ids.append(r.get("clip_id") or r.get("id"))
        return [i for i in ids if i]
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]



def resolve_beats(path=None):
    """Pretrained BEATs (iter3+, AS2M fine-tuned, cpt2). Only its config is needed to build the module; the weights
    come from the detector checkpoint. Downloaded from the public Hugging Face mirror when no local copy exists."""
    path = os.environ.get("BEATS_CKPT", path)
    if path and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download("WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2",
                           "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                           revision="172e4071a2912392d9151c70750b9a12f55fc54e")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--clip-list")
    ap.add_argument("--out", default="submission/predictions.jsonl")
    ap.add_argument("--zip", dest="zip_path", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--win-sec", type=float, default=10.0)
    ap.add_argument("--hop-sec", type=float, default=8.0)
    ap.add_argument("--save-probs", default=None,
                    help="npz of raw frame probabilities (for offline re-tuning)")
    # decoding overrides; default to the values tuned during training
    for k, t in [("high", float), ("low", float), ("median_k", int),
                 ("gap_sec", float), ("min_dur", float), ("expand_sec", float)]:
        ap.add_argument(f"--{k.replace('_','-')}", type=t, default=None)
    args = ap.parse_args()

    if not args.audio_dir and not args.manifest:
        raise SystemExit("need --audio-dir or --manifest")

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Architecture comes from the checkpoint, not a hardcoded default -- a
    # WavLM checkpoint loaded into a CRNN silently fails on state_dict keys.
    margs = ck.get("args", {}) or {}
    name = margs.get("model", "crnn")
    kw = {"ssl_name": margs.get("ssl_name", "microsoft/wavlm-base-plus")} \
        if name in ("ssl", "wavlm", "whisper") else {}
    if name == "beats_hybrid":
        kw = {
            "ssl_name": margs.get("ssl_name", "openai/whisper-medium"),
            "beats_ckpt": resolve_beats(margs.get("beats_ckpt"))
        }
    model = build_model(name, **kw).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"model: {name}")

    params = dict(ck.get("decode_params") or
                  {"high": 0.5, "low": 0.35, "median_k": 5, "gap_sec": 0.1,
                   "min_dur": 0.1, "expand_sec": 0.0})
    for k in list(params):
        v = getattr(args, k, None)
        if v is not None:
            params[k] = v
    print(f"checkpoint: {args.ckpt}")
    print(f"decode params: {params}")

    # collect (clip_id, path)
    items = []
    if args.audio_dir:
        for f in find_audio(args.audio_dir):
            items.append((os.path.splitext(os.path.basename(f))[0], f))
    else:
        root = os.path.dirname(os.path.abspath(args.manifest))
        with open(args.manifest) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    p = r["path"]
                    items.append((r["clip_id"],
                                  p if os.path.isabs(p) else os.path.join(root, p)))
    print(f"clips found: {len(items)}")

    preds, prob_store = {}, {}
    for i, (cid, path) in enumerate(items, 1):
        try:
            wav = read_audio(path)
        except Exception as e:                          # never lose a line
            print(f"  !! {cid}: read failed ({e}); emitting empty events")
            preds[cid] = []
            continue
        p, p_on, p_off, p_cat = frame_probs(model, wav, device, args.win_sec,
                                            args.hop_sec, args.amp)
        dec = {k: v for k, v in params.items()
               if k not in ("snap_win", "min_peak")}
        ev = decode_frames(p, FRAME_SEC, duration=len(wav) / SR, **dec)
        if params.get("snap_win"):
            ev = snap_events(ev, p_on, p_off, FRAME_SEC,
                             params["snap_win"], params.get("min_peak", 0.2),
                             len(wav) / SR)
        preds[cid] = ev
        if args.save_probs:
            # presence AND boundary tracks: an offline ensemble needs the
            # onset/offset heads too, or it silently loses the snapping gain.
            prob_store[cid] = p.astype(np.float16)
            prob_store[cid + "||on"] = p_on.astype(np.float16)
            prob_store[cid + "||off"] = p_off.astype(np.float16)
            prob_store[cid + "||cat"] = p_cat.astype(np.float16)
        if i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    # honour the authoritative clip list if given
    order = load_clip_list(args.clip_list)
    if order:
        missing = [c for c in order if c not in preds]
        extra = [c for c in preds if c not in set(order)]
        for c in missing:
            preds[c] = []
        print(f"clip-list: {len(order)} ids | missing audio: {len(missing)} "
              f"| dropped extras: {len(extra)}")
    else:
        order = sorted(preds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl_events(args.out, preds, clip_order=order)

    n_ev = sum(len(preds[c]) for c in order)
    n_empty = sum(1 for c in order if not preds[c])
    print(f"\nwrote {args.out}")
    print(f"  lines={len(order)}  events={n_ev}  "
          f"mean/clip={n_ev/max(1,len(order)):.2f}  empty clips={n_empty}")

    if args.save_probs:
        np.savez_compressed(args.save_probs, **prob_store)
        print(f"  probs -> {args.save_probs}")

    if args.zip_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.zip_path)), exist_ok=True)
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(args.out, "predictions.jsonl")      # must be at zip root
        print(f"  zip  -> {args.zip_path}")


if __name__ == "__main__":
    main()
