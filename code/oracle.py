"""Evaluation oracles for the campaign discrete-event simulation.

EmulatorOracle: Gaussian-process emulator (surrogate benchmark) of the FE
model, trained on the FE design-of-experiments at both fidelities.  It returns
the objectives, the strain at which the in-solver monitor would terminate the
analysis, and the solve time measured in the FE runs.
"""
import hashlib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from campaign import SC, to_max_form


def _pseudo_normal(q, lam, k):
    h = hashlib.sha256(('%s|%d|%d' % (np.round(q, 6).tolist(), lam, k)).encode()).digest()
    u1 = (int.from_bytes(h[:8], 'little') + 0.5) / 2 ** 64
    u2 = (int.from_bytes(h[8:16], 'little') + 0.5) / 2 ** 64
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


def _gp(X, y, noise=1e-2):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(X.shape[1], 0.5),
                                                   length_scale_bounds=(0.03, 20.0), nu=2.5) \
        + WhiteKernel(noise, (1e-6, 1.0))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(X, y)
    return gp


class EmulatorOracle:
    OUT = ('SEA', 'ip', 'logvar', 'smax', 'logwall')

    def __init__(self, records, limit, profiles, limit_lf=None):
        """records: list of dicts with q, lam, SEA, peak, plat_var, wall, eps_cross.
        profiles: {lam: (eps_grid, cumulative_wall_fraction)}"""
        self.limit = limit; self.profiles = profiles
        self.limits = {1: limit, 0: limit_lf if limit_lf is not None else limit}
        self.models = {}; self.noise = {}
        self.cross = {}
        for lam in (0, 1):
            R = [r for r in records if r['lam'] == lam]
            X = SC.fwd(np.array([r['q'] for r in R]))
            Rc = [r for r in R if r.get('clean', True)]
            Xc = SC.fwd(np.array([r['q'] for r in Rc]))
            Y = {'SEA': np.array([r['SEA'] for r in R]), 'ip': np.array([r['ip'] for r in R]),
                 'logvar': np.log10(np.maximum([r['plat_var'] for r in R], 1e-4)),
                 'smax': np.array([r['smax'] for r in R]),
                 'logwall': np.log(np.array([r['wall'] for r in R]))}
            for k in self.OUT:
                if lam == 1 and k not in ('logwall',):
                    # autoregressive (Kennedy-O'Hagan) correction on the LF emulator
                    m0 = self.models[(0, k)].predict(X)
                    A = np.column_stack([m0, np.ones_like(m0)])
                    coef, *_ = np.linalg.lstsq(A, Y[k], rcond=None)
                    resid = Y[k] - A @ coef
                    gp = _gp(X, resid)
                    self.models[(1, k)] = ('ar', coef, gp)
                elif k == 'logwall':
                    self.models[(lam, k)] = _gp(Xc, np.log(np.array([r['wall'] for r in Rc])))
                else:
                    self.models[(lam, k)] = _gp(X, Y[k])
            ec = np.array([r['eps_cross'] for r in R])
            fin = np.isfinite(ec)
            self.cross[lam] = _gp(X[fin], ec[fin]) if fin.sum() >= 4 else None
        for key, m in self.models.items():
            gp = m[2] if isinstance(m, tuple) else m
            self.noise[key] = float(np.sqrt(gp.kernel_.k2.noise_level) * gp._y_train_std)

    def predict(self, q, lam, k):
        x = SC.fwd(np.asarray(q))[None]
        m = self.models[(lam, k)]
        if isinstance(m, tuple):
            _, coef, gp = m
            m0 = self.models[(0, k)].predict(x)
            return float(coef[0] * m0[0] + coef[1] + gp.predict(x)[0])
        return float(m.predict(x)[0])

    def outputs(self, q, lam):
        out = {}
        for i, k in enumerate(self.OUT):
            out[k] = self.predict(q, lam, k) + self.noise[(lam, k)] * _pseudo_normal(q, lam, i)
        return out

    def evaluate(self, q, lam, prune=True):
        o = self.outputs(q, lam)
        full = float(np.exp(o['logwall']))
        y = to_max_form(o['SEA'], o['ip'], o['logvar'])
        infeasible = o['smax'] > self.limits[lam]
        if prune and infeasible:
            gp = self.cross[lam]
            ec = float(np.clip(gp.predict(SC.fwd(np.asarray(q))[None])[0], 0.02, 0.6)) if gp is not None else 0.3
            eg, frac = self.profiles[lam]
            dur = full * float(np.interp(ec, eg, frac))
            return dict(status='pruned', y=None, feasible=False, duration=dur, full_duration=full,
                        eps_end=ec, true_y=y, smax=o['smax'])
        return dict(status='completed', y=y, feasible=not infeasible, duration=full,
                    full_duration=full, eps_end=0.6, true_y=y, smax=o['smax'])
