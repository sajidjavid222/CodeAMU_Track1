"""One-at-a-time synthetic-branch scan from the v67 config, reported per cross-fit fold.

Accept rule (what made v67 transfer): a value must improve BOTH folds and sit on a
plateau, not a single-point spike.
"""
import json, sys, numpy as np
sys.path.insert(0, 'src')
from postprocess import decode_frames, split_events
from dur_snap import dur_snap
from metrics import score_submission
FRAME = 0.02
pr = json.load(open('work/route_v43.json'))['syn_params']
mem = json.load(open('work/syn_branch_v57.json'))['members']


def load(n):
    z = np.load(f'work/ovprobs/{n}.npz'); p, o, f = {}, {}, {}
    for k in z.files:
        if k.endswith('||on'): o[k[:-4]] = z[k].astype(np.float32)
        elif k.endswith('||off'): f[k[:-5]] = z[k].astype(np.float32)
        elif not k.endswith('||cat'): p[k] = z[k].astype(np.float32)
    return p, o, f


M = {n: load(n) for n in mem}
fold, recs = {}, []
for h in ('A', 'B'):
    for l in open(f'work/val_hold{h}.jsonl'):
        if l.strip():
            r = json.loads(l)
            if r.get('synthetic'): fold[r['clip_id']] = h; recs.append(r)
refs = {r['clip_id']: [(e['onset'], e['offset']) for e in r['events']] for r in recs}
durs = {r['clip_id']: r['duration'] for r in recs}
def resolve(h): return sorted({(m[:-2] + '_' + h if m.endswith(('_A', '_B')) and (m[:-2] + '_' + h) in M else m) for m in mem})
TR = {}
for c in refs:
    s = resolve(fold[c]); k = min(len(M[n][0][c]) for n in s)
    TR[c] = tuple(np.mean([M[n][i][c][:k] for n in s], axis=0).astype(np.float64) for i in range(3))

V67 = dict(high=pr['high'], low=pr['low'], median_k=pr['median_k'], gap_sec=pr['gap_sec'], min_dur=pr['min_dur'],
           expand_sec=pr['expand_sec'], st=0.06, ms=0.10, gp=0.40, split=0.45, ss=0.12, ps=0.10, sl=0.22, pl=0.20)


def run(P):
    out = {}
    for c in refs:
        p, on, off = TR[c]; d = durs[c]
        e = decode_frames(p, FRAME, duration=d, high=P['high'], low=P['low'], median_k=P['median_k'],
                          gap_sec=P['gap_sec'], min_dur=P['min_dur'], expand_sec=P['expand_sec'])
        e = split_events(e, on, off, FRAME, split_thr=P['st'], min_sub=P['ms'], max_pair_gap=P['gp'])
        out[c] = dur_snap(e, on, off, FRAME, duration=d, split=P['split'], snap_short=P['ss'],
                          peak_short=P['ps'], snap_long=P['sl'], peak_long=P['pl'])
    r = {}
    for h in ('A', 'B'):
        cs = [c for c in refs if fold[c] == h]
        r[h] = score_submission({c: out[c] for c in cs}, {c: refs[c] for c in cs}, {c: durs[c] for c in cs})['combined']
    r['all'] = score_submission(out, refs, durs)['combined']
    return r


if __name__ == '__main__':
    G = dict(high=[0.30, 0.35, 0.45, 0.50], low=[0.25, 0.30, 0.40], median_k=[1, 3, 7],
             gap_sec=[0.0, 0.02, 0.05, 0.10], min_dur=[0.06, 0.10, 0.15, 0.25],
             expand_sec=[0.0, 0.02, 0.03, 0.07, 0.10], st=[0.03, 0.04, 0.08, 0.10], ms=[0.06, 0.08, 0.12, 0.15],
             gp=[0.30, 0.35, 0.45, 0.50], split=[0.30, 0.38, 0.52, 0.60], ss=[0.06, 0.09, 0.15, 0.18],
             ps=[0.06, 0.08, 0.13, 0.16], pl=[0.10, 0.15, 0.25, 0.30])
    b = run(V67)
    print(f"v67 base  A={b['A']:.5f} B={b['B']:.5f} all={b['all']:.5f}   base params: "
          + json.dumps({k: V67[k] for k in G}), flush=True)
    for k, vals in G.items():
        line = []
        for v in vals:
            r = run(dict(V67, **{k: v}))
            dA, dB = r['A'] - b['A'], r['B'] - b['B']
            flag = '*' if dA > 1e-5 and dB > 1e-5 else ' '
            line.append(f"{v}:{dA:+.4f}/{dB:+.4f}{flag}")
        print(f"  {k:10} (base {V67[k]})  " + "  ".join(line), flush=True)
