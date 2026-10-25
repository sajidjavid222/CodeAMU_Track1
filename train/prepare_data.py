"""Convert the Vaani Noise Event parquet shards into wavs + a manifest.

The HF dataset (ARTPARK-IISc/Vaani-Noise-Event-Dataset) ships 182 parquet
shards with an embedded 16 kHz mono `audio` struct and a
`NoiseSubCategoryTimeStamp` list of {category, tag, start, end}.

This writes:
  <out>/audio/<clip_id>.wav
  <out>/manifest.jsonl   one record per clip:
      {clip_id, path, duration, tier, language, state, district,
       categories: [...], events: [{onset, offset, category, tag}, ...]}

Tier assignment ("gold" / "silver" / "bronze") drives the multi-tier loss in
train.py:
  * `annotationQuality` field when present (full dataset),
  * otherwise inferred: timestamps present -> silver, tags only -> bronze.

Usage:
    python src/prepare_data.py --parquet 'data/train/*.parquet' --out work/train
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

TARGET_SR = 16000

# The 7 top-level noise categories documented on the dataset card. Names are
# normalised because the preview shard uses `animal_sound` where the card says
# `animal`.
CATEGORIES = ["animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech"]
_CAT_ALIAS = {"animal_sound": "animal", "animals": "animal",
              "vehicle": "vehicle_traffic", "traffic": "vehicle_traffic",
              "baby": "baby_child", "child": "baby_child",
              "music": "singing_music", "singing": "singing_music",
              "phone": "phone_signal_alarm", "alarm": "phone_signal_alarm",
              "appliance": "appliance_machine", "machine": "appliance_machine",
              "human": "human_non_speech", "non_speech": "human_non_speech"}


def norm_category(c):
    """Map raw category strings onto the 7 canonical names.

    The shards are not consistent: the card says `animal` / `baby_child` but
    the data carries `animal_sound` / `baby_child_noise`, so strip the
    decorative suffixes before looking up the alias table.
    """
    if not c:
        return None
    c = str(c).strip().lower().replace(" ", "_").replace("-", "_")
    c = re.sub(r"_(noise|sound|sounds|noises)$", "", c)
    c = _CAT_ALIAS.get(c, c)
    if c in CATEGORIES:
        return c
    for canon in CATEGORIES:                # e.g. `vehicle_traffic_horn`
        if c.startswith(canon) or canon.startswith(c):
            return canon
    return c


def clip_id_from_path(p, fallback):
    """Filename stem is the natural clip id (matches Vaani wav naming)."""
    if not p:
        return fallback
    return re.sub(r"\.(wav|flac|mp3|ogg)$", "", os.path.basename(str(p)), flags=re.I)


# The dataset card's exact `annotationQuality` values.
_TIER_EXACT = {
    "verified_timestamps": "gold",      # ~11,111 segments / ~21.8 h
    "unverified_timestamps": "silver",  # ~61,642 segments / ~100.3 h
    "no_timestamps": "bronze",          # ~17,884 segments / ~32.4 h
}


def tier_of(row, events):
    """Map `annotationQuality` onto gold / silver / bronze.

    Substring order matters: "unverified_timestamps" contains "verified", so
    the negative case must be tested first or all 100 h of silver would be
    promoted to gold and its unreliable boundaries trusted at full weight.
    """
    q = (row.get("annotationQuality") or "").strip().lower()
    if q in _TIER_EXACT:
        return _TIER_EXACT[q]
    if q:                                        # tolerate renamed variants
        if "bronze" in q or "no_time" in q or "no_timestamp" in q:
            return "bronze"
        if "unverified" in q or "silver" in q or "noisy" in q:
            return "silver"
        if "verified" in q or "gold" in q:
            return "gold"
    return "silver" if events else "bronze"


# Transcript inline tags: <horn>, [breathing], <dog barking> ...
_TAG_RE = re.compile(r"[<\[]\s*/?\s*([a-zA-Z][a-zA-Z0-9 _-]{0,40})\s*[>\]]")
_NON_EVENT_TAGS = {"noise", "static noise", "unk", "unknown", "silence"}


def tags_from_transcript(t):
    if not t:
        return []
    out = []
    for m in _TAG_RE.finditer(str(t)):
        tag = m.group(1).strip().lower()
        if tag and tag not in _NON_EVENT_TAGS and tag not in out:
            out.append(tag)
    return out


def decode_audio(raw, sr=TARGET_SR):
    """Decode embedded audio bytes -> (mono float32 @ sr) or None.

    A handful of clips in the corpus are not readable by libsndfile (truncated
    or non-WAV payloads), so fall back to librosa/audioread and, failing that,
    report None. One bad clip out of 90k must not abort a 17.7 GB extraction.
    """
    if not raw:
        return None
    try:
        y, in_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:
        try:
            import librosa
            y, in_sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        except Exception:
            return None
    if y is None or len(y) == 0:
        return None
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if in_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    return y.astype("float32")


def process_shard(path, audio_dir, records, seen, write_audio=True, sr=TARGET_SR):
    table = pq.read_table(path)
    cols = set(table.schema.names)
    n_new = 0
    n_bad = 0

    for row in table.to_pylist():
        audio = row.get("audio") or {}
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        src_path = audio.get("path") if isinstance(audio, dict) else None
        clip_id = clip_id_from_path(src_path, f"clip_{len(records):07d}")
        if clip_id in seen:                        # dedupe across shards
            continue
        seen.add(clip_id)

        duration = float(row.get("duration") or 0.0)
        wav_path = os.path.join(audio_dir, f"{clip_id}.wav")

        if raw is not None and write_audio and not os.path.exists(wav_path):
            y = decode_audio(raw, sr)
            if y is None:
                seen.discard(clip_id)              # allow a retry from another shard
                n_bad += 1
                continue
            sf.write(wav_path, y, sr, subtype="PCM_16")
            if duration <= 0:
                duration = len(y) / sr

        events = []
        for ev in (row.get("NoiseSubCategoryTimeStamp") or []):
            if not isinstance(ev, dict):
                continue
            s, e = ev.get("start"), ev.get("end")
            if s is None or e is None:
                continue
            on, off = float(s), float(e)
            if off <= on:                          # drop degenerate spans
                continue
            if duration > 0:
                on, off = max(0.0, on), min(duration, off)
                if off <= on:
                    continue
            events.append({"onset": round(on, 4), "offset": round(off, 4),
                           "category": norm_category(ev.get("category")),
                           "tag": (ev.get("tag") or "").strip()})
        events.sort(key=lambda x: x["onset"])

        cats = [norm_category(c) for c in (row.get("NoiseCategory") or []) if c]
        rec = {
            "clip_id": clip_id,
            "path": os.path.relpath(wav_path, os.path.dirname(audio_dir)),
            "duration": round(duration, 4),
            "tier": tier_of(row, events),
            "language": row.get("language"),
            "state": row.get("state"),
            "district": row.get("district"),
            "categories": sorted({c for c in cats if c}),
            "events": events,
        }
        if "transcript" in cols:
            rec["weak_tags"] = tags_from_transcript(row.get("transcript"))
        records.append(rec)
        n_new += 1
    return n_new, n_bad


def _worker(job):
    """Process one shard in a subprocess -> its records (dedup happens later)."""
    shard, audio_dir, write_audio = job
    recs, seen = [], set()
    _, n_bad = process_shard(shard, audio_dir, recs, seen, write_audio=write_audio)
    return shard, recs, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="glob for parquet shards, e.g. 'data/train/*.parquet'")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-audio", action="store_true",
                    help="manifest only (labels/stats), skip wav extraction")
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel shard workers (90k clips is slow serially)")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.parquet))
    if args.limit_shards:
        shards = shards[:args.limit_shards]
    if not shards:
        raise SystemExit(f"no parquet files matched {args.parquet!r}")

    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    records, seen = [], set()
    total_bad = 0

    if args.jobs > 1:
        import multiprocessing as mp
        jobs = [(sh, audio_dir, not args.no_audio) for sh in shards]
        with mp.Pool(args.jobs) as pool:
            for i, (sh, recs, n_bad) in enumerate(pool.imap_unordered(_worker, jobs), 1):
                new = 0
                for r in recs:                      # global dedup at merge time
                    if r["clip_id"] in seen:
                        continue
                    seen.add(r["clip_id"])
                    records.append(r)
                    new += 1
                total_bad += n_bad
                print(f"[{i}/{len(shards)}] {os.path.basename(sh)}: +{new} clips "
                      f"(total {len(records)}, undecodable {total_bad})", flush=True)
        records.sort(key=lambda r: r["clip_id"])
    else:
        for i, sh in enumerate(shards, 1):
            n, n_bad = process_shard(sh, audio_dir, records, seen,
                                     write_audio=not args.no_audio)
            total_bad += n_bad
            print(f"[{i}/{len(shards)}] {os.path.basename(sh)}: +{n} clips "
                  f"(total {len(records)})", flush=True)

    man = os.path.join(args.out, "manifest.jsonl")
    with open(man, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    tiers, n_ev, dur = {}, 0, 0.0
    ev_durs, cat_count = [], {}
    for r in records:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        n_ev += len(r["events"])
        dur += r["duration"]
        for e in r["events"]:
            ev_durs.append(e["offset"] - e["onset"])
            cat_count[e["category"]] = cat_count.get(e["category"], 0) + 1

    print(f"\nwrote {man}")
    print(f"  clips={len(records)}  audio={dur/3600:.2f} h  events={n_ev}")
    print(f"  tiers={tiers}")
    if total_bad:
        print(f"  skipped {total_bad} undecodable clips")
    if ev_durs:
        a = np.array(ev_durs)
        print(f"  event dur: mean={a.mean():.2f}s median={np.median(a):.2f}s "
              f"p10={np.percentile(a,10):.2f}s p90={np.percentile(a,90):.2f}s")
        print(f"  categories={dict(sorted(cat_count.items(), key=lambda x:-x[1]))}")


if __name__ == "__main__":
    main()
