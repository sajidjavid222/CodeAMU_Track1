"""Train the multi-tier noise-event detector for Datathon@IndoML 2026 Track 1.

The competition's stated challenge is "effectively utilising data annotated at
different quality levels", so the loss is explicitly tier-aware:

    L = w_gold   * BCE_frame(gold rows)
      + w_silver * BCE_frame(silver rows)      (down-weighted: noisy timestamps)
      + w_cat    * BCE_frame_category(gold+silver)
      + w_weak   * BCE_clip(all rows)          (the only signal bronze provides)
      + w_cons   * mean-teacher consistency    (optional, --mean-teacher)

Silver is down-weighted rather than dropped: it is 100 of the 150 hours, so
discarding it wastes most of the corpus, but trusting its boundaries as much as
gold's degrades the very boundary precision Event-F1 measures.

Validation runs the real decoder and the real leaderboard metric, tunes the
decoding thresholds on the validation split, and checkpoints on Combined
(= Event F1 + Segment Dice), not on loss.

Example:
    python src/train.py --manifest work/train/manifest.jsonl --root work/train \
        --out runs/crnn_v1 --epochs 40 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics import score_submission
from postprocess import decode_all, decode_all2, tune, tune2
from sed_data import FRAME_SEC, VaaniSED, collate, load_manifest, split_manifest
from sed_model import build_model


def masked_bce(logit, target, mask, weight=None, frame_w=None):
    """BCE-with-logits averaged over valid frames only.

    `frame_w` is a per-frame multiplier used to silence individual frames --
    specifically the ones adjacent to an unverified (silver) boundary, whose
    label is not trustworthy even though the rest of the clip's is.
    """
    if mask.sum() == 0:
        return logit.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if loss.dim() == 3:                       # (B, F, C) category head
        m = mask.unsqueeze(-1).float()
    else:
        m = mask.float()
    if weight is not None:
        w = weight.view(-1, *([1] * (loss.dim() - 1)))
        loss = loss * w
        m = m * (w > 0).float()
    if frame_w is not None:
        fwv = frame_w[:, :loss.shape[1]]
        m = m * (fwv.unsqueeze(-1) if loss.dim() == 3 else fwv)
    denom = m.sum().clamp(min=1.0)
    return (loss * m).sum() / denom


@torch.no_grad()
def predict_probs(model, loader, device, amp=False):
    """Run the model over a loader -> per-clip frame probability tracks."""
    model.eval()
    probs, refs, durations = {}, {}, {}
    onsets, offsets = {}, {}
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        mask = batch["mask"].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(wav, frame_mask=mask)
        p = torch.sigmoid(out["strong_logit"].float()).cpu().numpy()
        # Frame counts differ slightly by backbone (CRNN 20.00 ms vs WavLM
        # 20.12 ms), so align the mask to what the model actually emitted
        # rather than assuming the two agree.
        on = torch.sigmoid(out["onset_logit"].float()).cpu().numpy()
        off = torch.sigmoid(out["offset_logit"].float()).cpu().numpy()
        nf = p.shape[1]
        m = batch["mask"].numpy()[:, :nf]
        for i, cid in enumerate(batch["clip_id"]):
            sel = m[i]
            probs[cid] = p[i][:m.shape[1]][sel]
            onsets[cid] = on[i][:m.shape[1]][sel]
            offsets[cid] = off[i][:m.shape[1]][sel]
            durations[cid] = batch["duration"][i]
    return probs, durations, onsets, offsets


def build_refs(records):
    return {r["clip_id"]: [(e["onset"], e["offset"]) for e in r.get("events", [])]
            for r in records}


def evaluate(model, loader, records, device, params=None, do_tune=True, amp=False):
    probs, durations, onsets, offsets = predict_probs(model, loader, device, amp)
    refs = build_refs(records)
    if do_tune:
        params, score = tune2(probs, refs, FRAME_SEC, durations,
                              onsets, offsets, verbose=False)
    else:
        params = params or {"high": 0.5, "low": 0.35, "median_k": 5,
                            "gap_sec": 0.1, "min_dur": 0.1, "expand_sec": 0.0}
        preds = decode_all2(probs, FRAME_SEC, durations, onsets, offsets, **params)
        score = score_submission(preds, refs, durations)
    return score, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True, help="dir that manifest paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-manifest", default=None, help="explicit val manifest")
    ap.add_argument("--model", default="crnn", choices=["crnn", "crnn_big", "ssl", "wavlm", "whisper", "beats_hybrid"])
    ap.add_argument("--ssl-name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--beats-ckpt", default=None)
    ap.add_argument("--lr-encoder", type=float, default=3e-5,
                    help="SSL encoder LR (much lower than the head's)")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=1,
                    help="warm up the head with the SSL encoder frozen")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # tier weights
    ap.add_argument("--w-gold", type=float, default=1.0)
    ap.add_argument("--w-silver", type=float, default=0.4)
    ap.add_argument("--w-cat", type=float, default=0.3)
    ap.add_argument("--w-weak", type=float, default=0.5)
    ap.add_argument("--w-bound", type=float, default=1.0,
                    help="weight on the Gaussian onset/offset heads")
    ap.add_argument("--silver-edge-sec", type=float, default=0.0,
                    help="ignore frames within +/- this many seconds of a "
                         "SILVER boundary (its edges are unverified)")
    ap.add_argument("--gold-only", action="store_true",
                    help="train on gold clips only (fine-tune phase)")
    ap.add_argument("--init-from", default=None,
                    help="initialise weights from a checkpoint (fine-tuning)")
    ap.add_argument("--mean-teacher", action="store_true")
    ap.add_argument("--w-cons", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #records")
    ap.add_argument("--swa-start", type=int, default=0,
                    help="epoch to begin weight averaging (0 = off). Best-epoch "
                         "selection on a 1.1k-clip val set picks noise spikes "
                         "(~+0.03 over the true plateau); an SWA average is a "
                         "single unselected candidate and generalises better.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/state.pt if it exists")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    records = load_manifest(args.manifest)
    if args.limit:
        records = records[:args.limit]
    if args.val_manifest:
        train_recs = records
        val_recs = load_manifest(args.val_manifest)
    else:
        train_recs, val_recs = split_manifest(records, args.val_frac, args.seed)

    if args.gold_only:
        n0 = len(train_recs)
        train_recs = [r for r in train_recs if r.get("tier") == "gold"]
        print(f"gold-only: {n0} -> {len(train_recs)} clips", flush=True)

    tiers = {}
    for r in train_recs:
        tiers[r.get("tier")] = tiers.get(r.get("tier"), 0) + 1
    print(f"train={len(train_recs)} val={len(val_recs)} tiers={tiers}", flush=True)

    tr_ds = VaaniSED(train_recs, args.root, args.max_dur, train=True,
                     silver_edge_sec=args.silver_edge_sec)
    va_ds = VaaniSED(val_recs, args.root, args.max_dur, train=False, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, collate_fn=collate,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, collate_fn=collate)

    kw = {"ssl_name": args.ssl_name} \
        if args.model in ("ssl", "wavlm", "whisper", "beats_hybrid") else {}
    if args.model == "beats_hybrid":
        kw["beats_ckpt"] = args.beats_ckpt
    model = build_model(args.model, **kw).to(device)
    print(f"model={args.model} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M",
          flush=True)

    if args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = st.get("model", st)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"initialised from {args.init_from} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    teacher = None
    if args.mean_teacher:
        # NOT deepcopy: BEATs contains weight_norm layers, which raise
        # "Only Tensors created explicitly by the user (graph leaves) support
        # the deepcopy protocol". Build a fresh model and copy the weights.
        teacher = build_model(args.model, **kw).to(device)
        teacher.load_state_dict(model.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

    if hasattr(model, "param_groups"):
        groups = model.param_groups(lr_head=args.lr, lr_encoder=args.lr_encoder)
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        max_lr = [g["lr"] for g in groups]
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        max_lr = args.lr
    steps = max(1, len(tr_ld)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr,
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # Encoder params that are trainable at construction time (i.e. excluding
    # the permanently-frozen conv feature extractor).
    encoder_names = {n for n, p in model.named_parameters()
                     if n.startswith("encoder.") and p.requires_grad}

    swa_state, swa_n = None, 0
    tier_w = {"gold": args.w_gold, "silver": args.w_silver, "bronze": 0.0}
    best = {"combined": -1.0}
    history = []
    start_epoch = 1

    # Resume: this cluster charges per job submission and caps wall time by
    # partition, so a run that hits the limit must not cost a whole retrain.
    state_path = os.path.join(args.out, "state.pt")
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])

        # OneCycleLR bakes total_steps into its state, so restoring it verbatim
        # only works when --epochs is unchanged (the wall-limit restart case).
        # If the schedule length changed, rebuild it and fast-forward instead of
        # dying with "Tried to step N times".
        sched_state = st.get("sched") or {}
        if sched_state.get("total_steps") == steps:
            sched.load_state_dict(sched_state)
        else:
            done = min((st.get("epoch", 0)) * max(1, len(tr_ld)), steps - 1)
            for _ in range(done):
                sched.step()
            print(f"  schedule length changed "
                  f"({sched_state.get('total_steps')} -> {steps} steps); "
                  f"rebuilt and fast-forwarded {done} steps", flush=True)
        if teacher is not None and st.get("teacher"):
            teacher.load_state_dict(st["teacher"])
        best = st.get("best", best)
        history = st.get("history", [])
        start_epoch = st.get("epoch", 0) + 1
        print(f"resumed from {state_path} at epoch {start_epoch} "
              f"(best combined={best.get('combined', -1):.4f})", flush=True)
        if start_epoch > args.epochs:
            print("already complete; nothing to do")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, agg, nb = time.time(), {}, 0

        # Head warm-up: a randomly-initialised head back-propagating into the
        # pretrained encoder destroys its features in the first few hundred
        # steps, so hold the encoder still until the head is sane. Only toggle
        # params that were trainable to begin with -- the conv feature
        # extractor is frozen for the whole run and is not in the optimizer.
        if encoder_names:
            frozen = epoch <= args.freeze_encoder_epochs
            for n_, p_ in model.named_parameters():
                if n_ in encoder_names:
                    p_.requires_grad_(not frozen)
            if frozen and epoch == 1:
                print(f"  encoder frozen for {args.freeze_encoder_epochs} "
                      f"warm-up epoch(s)", flush=True)
            elif epoch == args.freeze_encoder_epochs + 1:
                print("  encoder unfrozen", flush=True)

        for batch in tr_ld:
            wav = batch["wav"].to(device, non_blocking=True)
            mask = batch["mask"].to(device)
            frame = batch["frame"].to(device)
            frame_cat = batch["frame_cat"].to(device)
            weak = batch["weak"].to(device)
            weak_cat = batch["weak_cat"].to(device)
            w_row = torch.tensor([tier_w.get(t, 0.0) for t in batch["tier"]],
                                 device=device)

            with torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(wav, frame_mask=mask)
                nf = out["strong_logit"].shape[1]
                m, fr, frc = mask[:, :nf], frame[:, :nf], frame_cat[:, :nf]

                on_t = batch["onset_t"].to(device)[:, :nf]
                off_t = batch["offset_t"].to(device)[:, :nf]
                fw = batch["frame_w"].to(device)[:, :nf]
                # Boundary heads learn from GOLD only: teaching precise edges
                # from unverified silver timestamps is self-defeating, since
                # edge placement is exactly what the metric scores.
                w_gold_only = torch.tensor(
                    [args.w_gold if t == "gold" else 0.0 for t in batch["tier"]],
                    device=device)
                l_bound = masked_bce(out["onset_logit"], on_t, m, w_gold_only) \
                    + masked_bce(out["offset_logit"], off_t, m, w_gold_only)
                l_strong = masked_bce(out["strong_logit"], fr, m, w_row, fw)
                l_cat = masked_bce(out["cat_logit"], frc, m, w_row, fw)
                l_weak = F.binary_cross_entropy_with_logits(out["weak_logit"], weak) \
                    + F.binary_cross_entropy_with_logits(out["weak_cat_logit"], weak_cat)
                loss = l_strong + args.w_cat * l_cat + args.w_weak * l_weak \
                    + args.w_bound * l_bound

                l_cons = torch.zeros((), device=device)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(wav, frame_mask=mask)
                    l_cons = F.mse_loss(torch.sigmoid(out["strong_logit"]),
                                        torch.sigmoid(t_out["strong_logit"][:, :nf]))
                    ramp = min(1.0, epoch / max(1, args.epochs * 0.3))
                    loss = loss + args.w_cons * ramp * l_cons

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if teacher is not None:
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(), model.parameters()):
                        tp.mul_(args.ema_decay).add_(sp.detach(), alpha=1 - args.ema_decay)
                    for tb, sb in zip(teacher.buffers(), model.buffers()):
                        tb.copy_(sb)

            for k, v in [("loss", loss), ("strong", l_strong), ("cat", l_cat),
                         ("weak", l_weak), ("bound", l_bound), ("cons", l_cons)]:
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            nb += 1

        tr_line = " ".join(f"{k}={agg[k]/max(1,nb):.4f}" for k in agg)
        eval_model = teacher if teacher is not None else model
        score, params = evaluate(eval_model, va_ld, val_recs, device, amp=args.amp)

        # per-subset breakdown, mirroring the leaderboard's natural/synthetic split
        try:
            syn_ids = {r["clip_id"] for r in val_recs if r.get("synthetic")}
            if syn_ids and len(syn_ids) < len(val_recs):
                probs_s, durs_s, ons_s, offs_s = predict_probs(eval_model, va_ld,
                                                               device, args.amp)
                from postprocess import decode_all2 as _da2
                preds_s = _da2(probs_s, FRAME_SEC, durs_s, ons_s, offs_s, **params)
                for tag, keep in (("nat", lambda c: c not in syn_ids),
                                  ("syn", lambda c: c in syn_ids)):
                    sub_refs = {r["clip_id"]: [(e["onset"], e["offset"])
                                               for e in r.get("events", [])]
                                for r in val_recs if keep(r["clip_id"])}
                    sub_pred = {k: v for k, v in preds_s.items() if k in sub_refs}
                    ss = score_submission(sub_pred, sub_refs, durs_s)
                    print(f"      {tag}: F1={ss['event_f1']:.4f} "
                          f"Dice={ss['segment_dice']:.4f} comb={ss['combined']:.4f}",
                          flush=True)
        except Exception as e:
            print(f"      (subset report failed: {type(e).__name__})", flush=True)

        print(f"ep{epoch:03d} {tr_line} | val combined={score['combined']:.4f} "
              f"F1={score['event_f1']:.4f} Dice={score['segment_dice']:.4f} "
              f"P={score['precision']:.3f} R={score['recall']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        history.append({"epoch": epoch, **{k: v for k, v in score.items()},
                        "params": params})

        # ---- stochastic weight averaging over the plateau ----------------
        if args.swa_start and epoch >= args.swa_start:
            cur = {k: v.detach().float().cpu()
                   for k, v in eval_model.state_dict().items()}
            if swa_state is None:
                swa_state, swa_n = cur, 1
            else:
                swa_n += 1
                for k in swa_state:
                    if swa_state[k].is_floating_point():
                        swa_state[k].mul_(1 - 1.0 / swa_n).add_(cur[k], alpha=1.0 / swa_n)
                    else:
                        swa_state[k] = cur[k]
            swa_model = build_model(args.model, **kw).to(device)
            swa_model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
            s_score, s_params = evaluate(swa_model, va_ld, val_recs, device,
                                         amp=args.amp)
            print(f"      SWA(n={swa_n}) combined={s_score['combined']:.4f} "
                  f"F1={s_score['event_f1']:.4f} Dice={s_score['segment_dice']:.4f}",
                  flush=True)
            torch.save({"model": swa_model.state_dict(), "args": vars(args),
                        "decode_params": s_params, "score": dict(s_score),
                        "swa_n": swa_n, "frame_sec": FRAME_SEC},
                       os.path.join(args.out, "swa.pt"))
            with open(os.path.join(args.out, "swa.json"), "w") as fh:
                json.dump({**s_score, "swa_n": swa_n,
                           "decode_params": s_params}, fh, indent=2)
            del swa_model

        if score["combined"] > best["combined"]:
            best = dict(score)
            best["epoch"], best["decode_params"] = epoch, params
            torch.save({"model": eval_model.state_dict(), "args": vars(args),
                        "decode_params": params, "score": best,
                        "frame_sec": FRAME_SEC},
                       os.path.join(args.out, "best.pt"))
            with open(os.path.join(args.out, "best.json"), "w") as fh:
                json.dump(best, fh, indent=2)
            print(f"  -> saved best (combined={best['combined']:.4f})", flush=True)

        with open(os.path.join(args.out, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        # Full training state, rewritten atomically so a kill mid-write cannot
        # leave a corrupt checkpoint behind.
        tmp = state_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "teacher": teacher.state_dict() if teacher is not None else None,
                    "epoch": epoch, "best": best, "history": history,
                    "args": vars(args)}, tmp)
        os.replace(tmp, state_path)

    torch.save({"model": model.state_dict(), "args": vars(args),
                "frame_sec": FRAME_SEC}, os.path.join(args.out, "last.pt"))
    print(f"\nBEST combined={best['combined']:.4f} at epoch {best.get('epoch')}")
    print(f"decode params: {best.get('decode_params')}")


if __name__ == "__main__":
    main()
