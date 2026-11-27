"""Surrogate accuracy for ten random initialisations of the hyperparameter fit.

Fits the multi-task GPs of Fig. 11 with random seeds 0-9 and writes the
coefficient of determination on the independent test set to
data/parity_seeds.json (used for the ranges quoted in Section 8.1).
Takes under a minute.

Usage:
    python code/parity_seeds.py
"""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys
import json
sys.path.insert(0, _ROOT + 'code')
import fig_parity

DATA = _ROOT + 'data/'

if __name__ == '__main__':
    res = {}
    for seed in range(10):
        m = fig_parity.main(seed=seed, save=False)
        res[str(seed)] = {k: round(v['R2'], 3) for k, v in m.items()}
        print('seed %d: ' % seed + ', '.join('%s %.3f' % kv for kv in res[str(seed)].items()), flush=True)
    json.dump(res, open(DATA + 'parity_seeds.json', 'w'), indent=1)
    print('written data/parity_seeds.json')
