"""Collect FE results into records with all metrics."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _ROOT + 'code')
from metrics import curve_metrics, first_crossing, EPS_A

RUNS = _ROOT + 'runs/'
RHO_S = 930.0
LIMIT_LF = None


def load(prefix):
    d = dict(np.load(prefix + '_hist.npz'))
    m = json.load(open(prefix + '_meta.json'))
    return d, m


def record(prefix, limit=None):
    d, m = load(prefix)
    e, s = d['strain'], d['stress']
    met = curve_metrics(e, s, rho_eff=m['rho'] * RHO_S)
    ip = float(np.max(s[e <= EPS_A + 1e-12]))
    tt = e / e[-1]
    ke = d['KE'] / np.maximum(d['IE'], 1e-12)
    r = dict(name=os.path.relpath(prefix, RUNS), q=m['q'], N=m['N'], rho=m['rho'], s=m['s'],
             lam=1 if m['N'] == 36 else (0 if m['N'] == 24 else -1),
             SEA=met['SEA'], SEA_ed=met['SEA_ed'], Wv=met['Wv'], eps_d=met['eps_d'], ip=ip, smax=float(np.max(s)), peak_ed=met['peak'],
             plat_mean=met['plat_mean'], plat_var=met['plat_var'], W60=met['W60'],
             wall=float(m['wall']), steps=int(m['steps']), ne=int(m['ne']), status=m['status'],
             ke_ie_max=float(ke[tt >= 0.1].max()), hg_ie=float(d['HG'][-1] / d['IE'][-1]),
             added_mass=float(d['added_mass'][-1]), eroded=float(d['ninv'][-1]),
             lat_strain=float(d['Lx'][-1] / 25.0 - 1.0))
    if limit is not None:
        lim = limit if r['lam'] != 0 else LIMIT_LF
        r['eps_cross'] = first_crossing(e, s, lim)
        r['limit_used'] = lim
    return r


def collect(limit=None, patterns=('doe/*', 'test/*', 'conv/*', 'val/*', 'rate/*')):
    recs = []
    for pat in patterns:
        for f in sorted(glob.glob(RUNS + pat + '_meta.json')):
            pre = f[:-10]
            if os.path.exists(pre + '_hist.npz'):
                recs.append(record(pre, limit))
    return recs


def wall_profile(prefixes, grid=np.linspace(0, 0.6, 121)):
    P = []
    for pre in prefixes:
        d, m = load(pre)
        e = d['strain']; w = d['wall_hist']
        P.append(np.interp(grid, e, w / w[-1]))
    return grid, np.median(np.array(P), axis=0)


if __name__ == '__main__':
    lim = float(sys.argv[1]) if len(sys.argv) > 1 else None
    if len(sys.argv) > 2:
        LIMIT_LF = float(sys.argv[2])
    else:
        LIMIT_LF = lim
    recs = collect(lim)
    json.dump(recs, open(_ROOT + 'data/fe_records.json', 'w'), indent=1)
    for r in recs:
        print('%-18s N=%2d rho=%.3f SEA=%6.2f ip=%5.2f smax=%5.2f var=%.4f ed=%.3f wall=%5.0f ke=%.3f hg=%.3f am=%.3f lat=%.3f%s' % (
            r['name'], r['N'], r['rho'], r['SEA'], r['ip'], r['smax'], r['plat_var'], r['eps_d'], r['wall'],
            r['ke_ie_max'], r['hg_ie'], r['added_mass'], r['lat_strain'],
            (' ec=%.3f' % r['eps_cross']) if 'eps_cross' in r else ''))
