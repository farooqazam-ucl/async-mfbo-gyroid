"""Run optimisation campaigns (arms x seeds) against the FE emulator."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, os, time, pickle
import numpy as np
sys.path.insert(0, _ROOT + 'code')
from campaign import run_campaign, load_pool, SC, to_max_form
from oracle import EmulatorOracle
from prior_model import prior_objectives
from analyse_fe import wall_profile, RUNS

DATA = _ROOT + 'data/'
ARMS = {'A': dict(mode='A', prune=True, warm=True), 'B': dict(mode='B', prune=True, warm=True),
        'C': dict(mode='C', prune=True, warm=True), 'A0': dict(mode='A', prune=False, warm=True),
        'B0': dict(mode='B', prune=False, warm=True), 'Bc': dict(mode='B', prune=True, warm=False),
        # baselines representing established approaches (equal CPU budget of 30 high-fidelity analyses)
        'A0c': dict(mode='A', prune=False, warm=False),                      # synchronous multi-fidelity BO
        'S': dict(mode='A', prune=False, warm=False, policy='sf', budget=30),  # synchronous single-fidelity BO
        'Sb': dict(mode='B', prune=False, warm=False, policy='sf', budget=30), # asynchronous single-fidelity BO
        'R': dict(mode='B', prune=False, warm=False, policy='random', budget=30)}  # space-filling search


def build_oracle(settings):
    recs = json.load(open(DATA + 'fe_records.json'))
    recs = [r for r in recs if r['name'].startswith('doe/') and r['lam'] in (0, 1)]
    clean = json.load(open(DATA + 'clean_timing.json'))
    for r in recs:
        r['clean'] = bool(clean.get(r['name'][4:], False))
    import glob
    prof = {}
    for lam, pat in ((0, 'lf'), (1, 'hf')):
        clean = json.load(open(DATA + 'clean_timing.json'))
        pres = [f[:-9] for f in sorted(glob.glob(RUNS + 'doe/%s_*_hist.npz' % pat))]
        pres = [p for p in pres if 'wall_hist' in np.load(p + '_hist.npz').files and clean.get(os.path.basename(p), False)]
        prof[lam] = wall_profile(pres)
    return EmulatorOracle(recs, settings['sigma_limit'], prof, limit_lf=settings['sigma_limit_lf'])


def prior_data(pool, n=64, seed=123):
    idx = np.random.default_rng(seed).choice(len(pool), n, replace=False)
    P = [prior_objectives(q) for q in pool[idx]]
    Y = to_max_form([p['SEA'] for p in P], [p['ip'] for p in P],
                    [np.log10(max(p['plat_var'], 1e-4)) for p in P])
    return dict(X=SC.fwd(pool[idx]), Y=Y)


def main(arms, seeds, budget=100):
    settings = json.load(open(DATA + 'settings.json'))
    oracle_path = DATA + 'oracle.pkl'
    if os.path.exists(oracle_path):
        oracle = pickle.load(open(oracle_path, 'rb'))
    else:
        oracle = build_oracle(settings); pickle.dump(oracle, open(oracle_path, 'wb'))
    pool, _ = load_pool()
    prior = prior_data(pool)
    os.makedirs(DATA + 'campaigns', exist_ok=True)
    for arm in arms:
        for seed in seeds:
            out = DATA + 'campaigns/%s_s%d.json' % (arm, seed)
            if os.path.exists(out):
                continue
            t0 = time.time()
            kw = dict(ARMS[arm]); bud = kw.pop('budget', budget)
            r = run_campaign(oracle, W=4, budget=bud, seed=seed, ref=np.array(settings['ref']),
                             sea_target=settings['sea_target'], cost_ratio=settings['cost_ratio'],
                             pool=pool, prior_data=prior, **kw)
            r['arm'] = arm; r['seed'] = seed; r['real_time_s'] = time.time() - t0
            json.dump(r, open(out, 'w'), default=float)
            print('%s seed %d: T_wall=%.0f s cpu=%.2f h hv=%.3f n_hf=%d pruned=%d util=%.2f ttt=%.0f dec=%.2fs (%.0fs)' % (
                arm, seed, r['T_wall'], r['cpu_hours'], r['hv_final'], r['n_hf'], r['n_pruned'], r['utilisation'],
                r['ttt'], r['mean_decision_s'], time.time() - t0), flush=True)


if __name__ == '__main__':
    arms = sys.argv[1].split(',')
    seeds = [int(s) for s in sys.argv[2].split(',')]
    main(arms, seeds)
