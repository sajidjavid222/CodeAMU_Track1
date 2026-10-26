#!/usr/bin/env python3
"""CodeAMU - IndoML 2026 Track 1 (noise event detection): full inference pipeline.

    python inference/run_pipeline.py --audio_dir <test WAV folder> --run_dir <work folder> \
        --val_dir <official validation folder> [--checkpoints checkpoints] [--stages 1,2,3,4,5]

Reproduces submission v93_s7 (Event F1 0.7895, Segment Dice 0.8278, Combined 1.6173) on the Phase 2 test set.
Every stage skips work whose output already exists (resumable). Stages (all run with cwd = --run_dir, the scripts
read/write run_dir/work/... and run_dir/submission/...):

  1 detectors : 24 sound-event detectors -> 20 ms (10 ms for hr_*) frame probabilities  work/evalprobs/<m>.npz  [GPU]
  2 decode    : synthetic/natural router + natural / synthetic ensembles + decoding    submission/v67, routing,
                natural ensemble tracks (work/eval_syn_pred_fixed.json, work/ens_nat_eval_v60.npz)
  3 refnoise      : match every clip against the validation reference noise (synthetic refnoise, synthetic+natural refnoise,
                natural-routed clips)                                                 work/refnoise/*.json
  4 spans     : GLS refnoise subtraction on synthetic clips -> subtracted spans           work/refnoise_sub_*/
  5 refine    : v69 (refnoise snaps) -> v70 (natural refnoise merge) -> v71 -> v74 -> v76 -> v80 -> v93 (natural post-processing)
                -> s2 (energy-verified refnoise spans) -> noise meter (Track 2 enhancer) -> s3 (drop speech-only
                synthetic false alarms) -> s4/5 (refnoise spans in natural clips) -> s7 (fill empty clips)
Final output: <run_dir>/output/predictions.jsonl and submission.zip.

--track2_inputs (with stages 1,2): the event timestamps and routing the CodeAMU Track 2 system takes as input, as made for
the Track 2 Phase 2 test audio: combined audio + detector routing (route_combined.py), then the stage 2 decode under that
routing, run in <run_dir>/track2_inputs so the stage 3-5 artefacts stay untouched. Writes
<run_dir>/output/track2_inputs/{track1_predictions.jsonl, route.json}.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

DETECTORS = ["beats_full_gold", "bg_lg3_syn", "bg_lg3_syn_gold", "bg_sml_syn", "cb_big", "cb_s7", "cb_s7_gold", "gv_lg3_A",
             "gv_lg3_B", "gv_med_A", "gv_med_B", "gv_sml_A", "gv_sml_B", "hr_med_syn", "hr_med_syn_gold", "lg2_full_gold",
             "lg3_full_gold", "ov_med_syn", "rb_beats_gold", "v2_crnn_gold", "v4_wavlm_gold", "v8_whis_lg_gold",
             "v9_lg2_gold", "v9_lg3_s21"]
ASSETS = ["synth_clf.pkl", "boundary_regressor.pkl", "route_v43.json", "syn_branch_v57.json", "detector_router.pkl"]
GLS_TAG = "gate0.5_gls_grow_th0.25_white2"
REFINE = [("v69", ["build_v68.py", "v69"]), ("v70", ["build_v68.py", "v70"]), ("v71", ["build_v71.py"]),
          ("v74", ["build_v74.py"]), ("v76", ["build_v76.py"]), ("v80", ["build_v80.py"]), ("v93", ["build_v93.py"]),
          ("v93_s2", ["build_v93_s2.py", "submission/v93"]), ("probe", ["t1_syn_fp_probe.py"]),
          ("v93_s3", ["build_v93_s3.py"]), ("v93_s4", ["build_v93_s4.py"]),
          ("v93_s5", ["build_v93_s5.py"]), ("v93_s7", ["build_v93_s7.py", "0.20"])]


def run(cmd, env, cwd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--val_dir", required=True, help="official validation folder (reference noise source)")
    ap.add_argument("--checkpoints", default=f"{ROOT}/checkpoints")
    ap.add_argument("--stages", default="1,2,3,4,5")
    ap.add_argument("--track2_inputs", action="store_true", help="also write the CodeAMU Track 2 inputs (after stage 2)")
    a = ap.parse_args()
    stages = {int(s) for s in a.stages.split(",")}
    rd = os.path.abspath(a.run_dir); audio = os.path.abspath(a.audio_dir); ck = os.path.abspath(a.checkpoints)
    for d in ("work/evalprobs", "work/refnoise", "submission", "output", "data"):
        os.makedirs(f"{rd}/{d}", exist_ok=True)
    if not os.path.exists(f"{rd}/data/validation"):
        os.symlink(os.path.abspath(a.val_dir), f"{rd}/data/validation")
    for f in ASSETS:
        shutil.copy(f"{ROOT}/assets/{f}", f"{rd}/work/{f}")
    ids = sorted(f[:-4] for f in os.listdir(audio) if f.lower().endswith(".wav"))
    with open(f"{rd}/work/eval_clip_ids.txt", "w") as fh:
        fh.write("\n".join(ids) + "\n")
    env = dict(os.environ, PYTHONPATH=HERE, T1_TEST_AUDIO=audio, T1_ENHANCER_DIR=f"{ck}/enhancer", PYTHONUNBUFFERED="1")
    print(f"{len(ids)} clips | run dir {rd}", flush=True)

    if 1 in stages:
        for m in DETECTORS:
            out = f"{rd}/work/evalprobs/{m}.npz"
            if os.path.exists(out):
                continue
            e = dict(env, SED_TIME_POOL="1" if m.startswith("hr_") else "2")     # hr_* detectors use a 10 ms grid
            run([PY, f"{HERE}/infer.py", "--ckpt", f"{ck}/detectors/{m}/best.pt", "--audio-dir", audio,
                 "--clip-list", f"{rd}/work/eval_clip_ids.txt", "--amp", "--out", f"{rd}/work/evalprobs/{m}.jsonl",
                 "--save-probs", f"{rd}/work/evalprobs/{m}.part.npz"], e, rd)
            os.replace(f"{rd}/work/evalprobs/{m}.part.npz", out)
    if 2 in stages and not os.path.exists(f"{rd}/submission/v67/predictions.jsonl"):
        run([PY, f"{HERE}/t1_v67_decode.py", audio, "work/eval_clip_ids.txt", "work/evalprobs", "submission/v67"], env, rd)
    if a.track2_inputs:
        t2 = f"{rd}/track2_inputs"
        os.makedirs(f"{t2}/work", exist_ok=True)
        os.makedirs(f"{rd}/output/track2_inputs", exist_ok=True)
        for f in ASSETS:
            shutil.copy(f"{ROOT}/assets/{f}", f"{t2}/work/{f}")
        if not os.path.exists(f"{rd}/work/route_combined.json"):
            run([PY, f"{HERE}/route_combined.py"], env, rd)
        if not os.path.exists(f"{t2}/v67r/predictions.jsonl"):
            run([PY, f"{HERE}/t1_v67_decode.py", audio, f"{rd}/work/eval_clip_ids.txt", f"{rd}/work/evalprobs", f"{t2}/v67r"],
                dict(env, T1_ROUTE_JSON=f"{rd}/work/route_combined.json"), t2)
        shutil.copy(f"{t2}/v67r/predictions.jsonl", f"{rd}/output/track2_inputs/track1_predictions.jsonl")
        shutil.copy(f"{rd}/work/route_combined.json", f"{rd}/output/track2_inputs/route.json")
        print(f"Track 2 inputs: {rd}/output/track2_inputs/", flush=True)
    if 3 in stages:
        for name, opts in (("test_pe", ["test", "pe"]), ("test_pe_nb", ["test", "pe", "nb"]), ("testnat_pe", ["testnat", "pe"])):
            if not os.path.exists(f"{rd}/work/refnoise/{name}.json"):
                run([PY, f"{HERE}/refnoise_match.py"] + opts, env, rd)
    if 4 in stages and not os.path.exists(f"{rd}/work/refnoise_sub_{GLS_TAG}/subtracted_spans.json"):
        run([PY, f"{HERE}/t2_refnoise_subtract.py", "test", "white2", "th0.25", "gls", "grow", "gate0.5"], env, rd)
    if 5 in stages:
        for tag, cmd in REFINE:
            done = f"{rd}/work/t1_syn_unsupported_D.json" if tag == "probe" else f"{rd}/submission/{tag}/predictions.jsonl"
            if not os.path.exists(done):
                run([PY, f"{HERE}/{cmd[0]}"] + cmd[1:], env, rd)
        for f in ("predictions.jsonl", "submission.zip"):
            shutil.copy(f"{rd}/submission/v93_s7/{f}", f"{rd}/output/{f}")
        print(f"final: {rd}/output/submission.zip", flush=True)


if __name__ == "__main__":
    main()
