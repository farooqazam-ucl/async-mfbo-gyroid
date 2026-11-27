"""Design-of-experiments generation for the FE campaign."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import json, sys
import numpy as np
from scipy.stats import qmc
sys.path.insert(0, _ROOT + 'code')
from geometry import ContinuumDensity, bezier_min

QLO, QHI = 0.1, 2.0
RHO = 0.25
TMIN = 1.2
_cd = None


def cdens():
    global _cd
    if _cd is None:
        _cd = ContinuumDensity()
    return _cd


def normalise(q):
    q = np.asarray(q, float)
    return q / q.sum(axis=-1, keepdims=True) * 4.0   # mean-one control vector


def tmin_of(q):
    return cdens().tmin(q, RHO)


def feasible_sobol(n, seed, skip_uniform=False):
    sob = qmc.Sobol(d=4, scramble=True, seed=seed)
    out = []; tm = []
    while len(out) < n:
        X = QLO + (QHI - QLO) * sob.random(64)
        for q in X:
            qn = normalise(q)
            t = tmin_of(qn)
            if t >= TMIN:
                out.append(qn); tm.append(t)
                if len(out) >= n:
                    break
    return np.array(out), np.array(tm)


if __name__ == '__main__':
    lf, tlf = feasible_sobol(48, seed=11)
    lf = np.vstack([[1, 1, 1, 1], lf[:47]]); tlf = np.concatenate([[tmin_of([1, 1, 1, 1])], tlf[:47]])
    test, tts = feasible_sobol(16, seed=2027)
    json.dump(dict(lf=lf.tolist(), tmin_lf=tlf.tolist(), test=test.tolist(), tmin_test=tts.tolist()),
              open(_ROOT + 'data/doe_designs.json', 'w'), indent=1)
    print('LF designs', lf.shape, 'tmin range %.2f-%.2f' % (tlf.min(), tlf.max()))
    print('test designs', test.shape, 'tmin range %.2f-%.2f' % (tts.min(), tts.max()))
