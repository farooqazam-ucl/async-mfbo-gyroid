import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, time, numpy as np
sys.path.insert(0, _ROOT + 'code')
from campaign import run_campaign, load_pool, SC, to_max_form
from prior_model import prior_objectives

class FakeOracle:
    def __init__(self, limit=7.0):
        self.limit = limit
    def evaluate(self, q, lam, prune=True):
        m = prior_objectives(q)
        bias = 1.0 if lam == 1 else 0.93
        sea = m['SEA'] * bias; peak = m['peak'] * bias; lv = np.log10(max(m['plat_var'], 1e-3))
        full = (450.0 if lam == 1 else 90.0) * (1 + 0.1 * np.sin(13 * q[0]))
        y = to_max_form(sea, m['plat_mean'] * 0 + peak, lv)
        if prune and peak > self.limit:
            return dict(status='pruned', y=None, feasible=False, duration=0.4 * full, full_duration=full, eps_end=0.25, true_y=y)
        return dict(status='completed', y=y, feasible=peak <= self.limit, duration=full, full_duration=full, eps_end=0.6, true_y=y)

if __name__ == "__main__":
    pool, tm = load_pool()
    print('pool', pool.shape)
    prior_idx = np.random.default_rng(1).choice(len(pool), 48, replace=False)
    P = [prior_objectives(q) for q in pool[prior_idx]]
    prior = dict(X=SC.fwd(pool[prior_idx]), Y=to_max_form([p['SEA'] for p in P], [p['peak'] for p in P], [np.log10(max(p['plat_var'], 1e-3)) for p in P]))
    ref = np.array([10.0, -7.0, -0.0])
    for mode in [m for m in sys.argv[1:] if not m.isdigit()]:
        t0 = time.time()
        r = run_campaign(FakeOracle(), mode=mode, W=4, budget=int(sys.argv[-1]) if sys.argv[-1].isdigit() else 16, seed=0, ref=ref, sea_target=13.0, pool=pool, prior_data=prior)
        print(mode, 'T_wall=%.0f cpu_h=%.3f hv=%.3f n_hf=%d pruned=%d util=%.2f dec=%.2fs ttt=%.0f  (%.0fs)' % (r['T_wall'], r['cpu_hours'], r['hv_final'], r['n_hf'], r['n_pruned'], r['utilisation'], r['mean_decision_s'], r['ttt'], time.time()-t0), flush=True)
        print('   lam seq:', ''.join(str(o['lam']) for o in r['obs']), ' rho:', [round(l['rho'],2) for l in r['log']][:40])
