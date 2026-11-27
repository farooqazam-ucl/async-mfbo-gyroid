"""Per-arm statistics of the optimisation campaigns."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, glob
import numpy as np
from scipy.stats import mannwhitneyu
DATA = _ROOT + 'data/'


def stats():
    R = {}
    for f in sorted(glob.glob(DATA + 'campaigns/*.json')):
        r = json.load(open(f)); R.setdefault(r['arm'], []).append(r)
    # allocated worker time: max(capacity, busy) integrated over the campaign
    for a, rs in R.items():
        for r in rs:
            occ = np.array([[o[1], o[2]] for o in r['occupancy']])
            tg = np.linspace(0, r['T_wall'], 20001)
            busy = ((occ[:, 0][None, :] <= tg[:, None]) & (occ[:, 1][None, :] > tg[:, None])).sum(1)
            if r['mode'] == 'C':
                wt = np.array(r['W_trace'])
                cap = wt[np.searchsorted(wt[:, 0], tg, side='right') - 1, 1]
            else:
                cap = np.full_like(tg, 4.0)
            alloc = np.maximum(cap, busy)
            r['alloc_worker_h'] = float(np.trapezoid(alloc, tg) / 3600.0)
            r['utilisation'] = float(np.trapezoid(busy, tg) / np.trapezoid(alloc, tg))
            r['mean_pool'] = float(np.trapezoid(alloc, tg) / r['T_wall'])
    keys = ['T_wall', 'cpu_hours', 'hv_final', 'ttt', 'n_pruned', 'cpu_saved_h', 'cpu_saved_est_h', 'utilisation',
            'n_hf', 'mean_decision_s', 'alloc_worker_h', 'mean_pool']
    S = {}
    for a, rs in R.items():
        S[a] = {}
        for k in keys:
            v = np.array([r[k] for r in rs], float)
            if k in ('T_wall', 'ttt'):
                v = v / 3600.0
            fin = v[np.isfinite(v)]
            S[a][k] = dict(med=float(np.median(fin)) if fin.size else float('inf'),
                           q1=float(np.percentile(fin, 25)) if fin.size else float('inf'),
                           q3=float(np.percentile(fin, 75)) if fin.size else float('inf'),
                           n_finite=int(fin.size), vals=v.tolist())
        S[a]['n'] = len(rs)
    tests = {}
    pairs = [('A', 'B'), ('A0', 'B0'), ('B', 'B0'), ('A', 'A0'), ('B', 'C'), ('B', 'Bc'),
             ('B', 'A0c'), ('B', 'S'), ('B', 'Sb'), ('B', 'R'), ('A0c', 'S'), ('S', 'R'), ('Sb', 'R')]
    for a, b in pairs:
        if a in R and b in R:
            tests['%s_vs_%s' % (a, b)] = {}
            for k in ('T_wall', 'cpu_hours', 'hv_final', 'ttt', 'utilisation'):
                x = np.array([r[k] for r in R[a]], float); y = np.array([r[k] for r in R[b]], float)
                x = np.where(np.isfinite(x), x, 1e12); y = np.where(np.isfinite(y), y, 1e12)
                p = mannwhitneyu(x, y, alternative='two-sided').pvalue
                tests['%s_vs_%s' % (a, b)][k] = float(p)
    out = dict(stats=S, tests=tests)
    json.dump(out, open(DATA + 'campaign_stats.json', 'w'), indent=1)
    return out


if __name__ == '__main__':
    out = stats()
    for a, s in out['stats'].items():
        print(a, 'n=%d' % s['n'], ' '.join('%s=%.3g[%.3g,%.3g]' % (k, s[k]['med'], s[k]['q1'], s[k]['q3'])
                                           for k in ('T_wall', 'cpu_hours', 'hv_final', 'ttt', 'n_pruned', 'cpu_saved_h', 'utilisation', 'n_hf')))
    for k, v in out['tests'].items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
