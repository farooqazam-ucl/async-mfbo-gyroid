"""Asynchronous multi-fidelity Bayesian optimisation with in-solver pruning:
surrogate models, acquisition, feasibility classifier and a discrete-event
simulation (DES) of the worker pool for execution modes A, B and C.

Objectives are handled in maximisation form for BoTorch:
    y = [SEA (kJ/kg), -sigma_peak (MPa), -log10(sigma_plat / MPa^2)].
"""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import json, math, os, sys, time, warnings, heapq
import numpy as np
import torch
warnings.filterwarnings('ignore')
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from botorch.models import MultiTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
import gpytorch
sys.path.insert(0, _ROOT + 'code')

TASK_PRIOR, TASK_LF, TASK_HF = 0, 1, 2


# ---------------------------------------------------------------------------
# candidate pool (feasible w.r.t. the analytic minimum-thickness constraint)
# ---------------------------------------------------------------------------
def load_pool(path=_ROOT + 'data/pool.npz', n=6000):
    if os.path.exists(path):
        d = np.load(path)
        return d['X'], d['tmin']
    from doe import feasible_sobol
    X, tm = feasible_sobol(n, seed=99)
    np.savez(path, X=X, tmin=tm)
    return X, tm


class Scaler:
    """Maps mean-one control vectors to [0,1]^4 using fixed bounds."""
    lo = np.array([0.25, 0.25, 0.25, 0.25]); hi = np.array([2.2, 2.2, 2.2, 2.2])

    def fwd(self, X):
        return (np.asarray(X) - self.lo) / (self.hi - self.lo)


SC = Scaler()


# ---------------------------------------------------------------------------
# hypervolume utilities
# ---------------------------------------------------------------------------
def hypervolume(Y, ref):
    """Y: (n,3) maximisation-form objectives; ref: (3,)"""
    if len(Y) == 0:
        return 0.0
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64)
    rt = torch.as_tensor(np.asarray(ref), dtype=torch.float64)
    Yt = Yt[(Yt > rt).all(dim=-1)]
    if Yt.shape[0] == 0:
        return 0.0
    bd = DominatedPartitioning(ref_point=rt, Y=Yt)
    return float(bd.compute_hypervolume())


def to_max_form(sea, peak, logvar):
    return np.stack([np.asarray(sea), -np.asarray(peak), -np.asarray(logvar)], axis=-1)


# ---------------------------------------------------------------------------
# feasibility classifier (variational GP, Bernoulli likelihood)
# ---------------------------------------------------------------------------
class GPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, Z):
        vd = gpytorch.variational.CholeskyVariationalDistribution(Z.size(0))
        vs = gpytorch.variational.VariationalStrategy(self, Z, vd, learn_inducing_locations=False)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=Z.size(-1)))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_classifier(X, ok, iters=None):
    """Binary GP classifier (Bernoulli likelihood, Laplace approximation)."""
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import Matern as SkMatern, ConstantKernel as SkC
    k = SkC(1.0, (1e-2, 1e2)) * SkMatern(length_scale=np.full(X.shape[1], 0.3),
                                         length_scale_bounds=(0.03, 10.0), nu=2.5)
    clf = GaussianProcessClassifier(k, n_restarts_optimizer=0, random_state=0)
    clf.fit(X, ok.astype(int))
    return clf


def prob_feasible(clf, Xc):
    if clf is None:
        return np.ones(len(Xc))
    return clf.predict_proba(np.asarray(Xc))[:, 1]


# ---------------------------------------------------------------------------
# multi-task surrogate
# ---------------------------------------------------------------------------
def build_mtgp(X, T, y, state=None, fit=True):
    Xt = torch.as_tensor(np.column_stack([X, T]))
    yt = torch.as_tensor(y).reshape(-1, 1)
    model = MultiTaskGP(Xt, yt, task_feature=-1, output_tasks=[TASK_HF],
                        covar_module=MaternKernel(nu=2.5, ard_num_dims=X.shape[1]),
                        outcome_transform=Standardize(m=1), all_tasks=[TASK_PRIOR, TASK_LF, TASK_HF])
    if state is not None:
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            state = None
    if fit or state is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll, optimizer_kwargs=dict(options=dict(maxiter=80)))
        except Exception:
            pass
    model.eval()
    return model


def lf_hf_correlation(model, Xc):
    """Posterior correlation between LF and HF predictions at each x."""
    n = len(Xc)
    rho = np.zeros(n)
    with torch.no_grad():
        for i in range(n):
            Xp = torch.as_tensor(np.array([[*Xc[i], TASK_LF], [*Xc[i], TASK_HF]]))
            p = model.posterior(Xp)
            C = p.mvn.covariance_matrix.reshape(2, 2) if p.mvn.covariance_matrix.numel() == 4 else p.mvn.covariance_matrix[..., :2, :2]
            C = C.reshape(2, 2)
            rho[i] = float(C[0, 1] / torch.sqrt(C[0, 0] * C[1, 1]).clamp_min(1e-12))
    return rho


def task_corr(model, i=TASK_LF, j=TASK_HF):
    try:
        C = model.covar_module.kernels[1].covar_matrix.to_dense().detach().numpy()
        return float(C[i, j] / np.sqrt(C[i, i] * C[j, j]))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# optimiser state and decision rule
# ---------------------------------------------------------------------------
class Optimiser:
    def __init__(self, pool, ref, cost_ratio, prior_data=None, rng=None, refit_every=5,
                 nmc=32, ncand=512, confirm_every=6, policy='mf'):
        # policy: 'mf' screen-and-confirm multi-fidelity (this work); 'sf' single-fidelity
        # (high fidelity only); 'random' space-filling high-fidelity designs (no model)
        self.policy = policy
        self.pool = pool; self.Xpool = SC.fwd(pool)
        self.ref = np.asarray(ref); self.cost_ratio = cost_ratio
        self.prior = prior_data  # dict X (scaled), Y (n,3)
        self.rng = rng or np.random.default_rng(0)
        self.refit_every = refit_every; self.nmc = nmc; self.ncand = ncand
        self.states = [None, None, None]; self.n_since_fit = 10 ** 6
        self.models = None; self.clf = None
        self.confirm_every = confirm_every

    def _training(self, obs):
        X, T, Y = [], [], []
        if self.prior is not None:
            X.append(self.prior['X']); T.append(np.full(len(self.prior['X']), TASK_PRIOR)); Y.append(self.prior['Y'])
        comp = [o for o in obs if o['status'] == 'completed']
        if comp:
            X.append(np.array([SC.fwd(o['q']) for o in comp]))
            T.append(np.array([TASK_HF if o['lam'] == 1 else TASK_LF for o in comp]))
            Y.append(np.array([o['y'] for o in comp]))
        return np.vstack(X), np.concatenate(T), np.vstack(Y)

    def update(self, obs):
        if self.policy == 'random':
            self.models = None
            return
        if self.prior is None and not any(o['status'] == 'completed' for o in obs):
            self.models = None
            return
        X, T, Y = self._training(obs)
        fit = self.n_since_fit >= self.refit_every or self.models is None
        models = []
        for j in range(3):
            m = build_mtgp(X, T, Y[:, j], state=self.states[j], fit=fit)
            if fit:
                self.states[j] = {k: v.clone() for k, v in m.state_dict().items()
                                  if 'train' not in k}
            models.append(m)
        self.n_since_fit = 0 if fit else self.n_since_fit + 1
        self.models = models
        # feasibility classifier over all FE evaluations
        fe = [o for o in obs if o['status'] in ('completed', 'pruned')]
        labels = np.array([1.0 if o['feasible'] else 0.0 for o in fe])
        if len(fe) >= 4 and labels.min() < 0.5 < labels.max():
            if getattr(self, '_clf_n', -1) != len(fe):
                self.clf = fit_classifier(np.array([SC.fwd(o['q']) for o in fe]), labels)
                self._clf_n = len(fe)
        else:
            self.clf = None
        hf = [SC.fwd(o['q']) for o in obs if o['status'] == 'completed' and o['lam'] == 1]
        self.obs_X = np.array(hf) if hf else None

    def candidates(self, obs):
        from doe import cdens, normalise, TMIN, RHO
        idx = self.rng.choice(len(self.pool), size=min(self.ncand, len(self.pool)), replace=False)
        C = [self.pool[idx]]
        comp = [o for o in obs if o['status'] == 'completed']
        good = [o for o in comp if o['feasible']]
        if good:
            Yg = np.array([o['y'] for o in good])
            from botorch.utils.multi_objective.pareto import is_non_dominated
            nd = is_non_dominated(torch.as_tensor(Yg)).numpy()
            base = np.array([good[i]['q'] for i in np.nonzero(nd)[0]])
            B = base[self.rng.integers(len(base), size=96)]
            L = normalise(np.clip(B * np.exp(0.08 * self.rng.standard_normal(B.shape)), 0.05, None))
            ok = cdens().tmin_batch(L, RHO) >= TMIN
            if ok.any():
                C.append(L[ok])
        # LF-screened designs not yet confirmed at high fidelity
        hfq = [tuple(np.round(o['q'], 8)) for o in obs if o['lam'] == 1]
        lfo = [o['q'] for o in comp if o['lam'] == 0 and o['feasible'] and tuple(np.round(o['q'], 8)) not in hfq]
        if lfo:
            C.append(np.array(lfo))
        return np.vstack(C)

    def score(self, Cq, pending_q):
        Xc = SC.fwd(Cq)
        model = ModelListGP(*self.models)
        Xp = torch.as_tensor(SC.fwd(np.array(pending_q))) if len(pending_q) else None
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.nmc]))
        if self.obs_X is not None:
            acq = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=torch.as_tensor(self.ref), X_baseline=torch.as_tensor(self.obs_X),
                X_pending=Xp, sampler=sampler, prune_baseline=True, cache_root=False)
        else:
            # no confirmed (high-fidelity) design yet: improvement over the reference point
            from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
            from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
            part = NondominatedPartitioning(ref_point=torch.as_tensor(self.ref), Y=torch.empty(0, 3))
            acq = qLogExpectedHypervolumeImprovement(model=model, ref_point=torch.as_tensor(self.ref),
                                                     partitioning=part, sampler=sampler, X_pending=Xp)
        vals = []
        with torch.no_grad():
            for i in range(0, len(Xc), 256):
                vals.append(acq(torch.as_tensor(Xc[i:i + 256]).unsqueeze(-2)).numpy())
        la = np.concatenate(vals)
        pf = prob_feasible(self.clf, Xc)
        return la + np.log(np.clip(pf, 1e-6, 1.0)), la, pf

    def propose(self, obs, pending, elastic=False):
        pending = list(pending)
        if self.models is None:
            # random policy, or no completed analysis yet (cold start): space-filling design
            q = self.pool[self.rng.integers(len(self.pool))]
            return q, (0 if self.policy == 'mf' else 1), dict(rho=0.0, logacq=0.0, pof=1.0, n_promising=1)
        """Return (q, lam, info).  pending: list of q currently running."""
        Cq = self.candidates(obs)
        tot, la, pf = self.score(Cq, pending)
        # multi-fidelity choice (augmented-EI cost weighting over the top-K
        # candidates): score(x, LF) = score(x, HF) + log(rho(x) * C1/C0)
        # screened (LF, feasible) designs that have not been confirmed at HF
        hfq = set(tuple(np.round(o['q'], 8)) for o in obs if o['lam'] == 1)
        hfq |= set(tuple(np.round(p, 8)) for p in pending)
        scr = [k for k, c in enumerate(Cq) if any(o['lam'] == 0 and o['status'] == 'completed' and o['feasible']
                                                  and np.allclose(o['q'], c) for o in obs)
               and tuple(np.round(c, 8)) not in hfq]
        scr_set = set(scr)
        # screen-and-confirm schedule: one dispatch in `confirm_every` confirms the
        # most promising screened design at high fidelity; the others screen the
        # best new candidate at low fidelity
        self.n_proposed = getattr(self, 'n_proposed', 0) + 1
        rho = float(np.mean([task_corr(m) for m in self.models]))
        if self.policy == 'sf':
            i = int(np.argmax(tot)); lam = 1
        elif scr and self.n_proposed % self.confirm_every == 0:
            i = scr[int(np.argmax(tot[scr]))]; lam = 1
        else:
            new_idx = [k for k in range(len(Cq)) if k not in scr_set]
            i = new_idx[int(np.argmax(tot[new_idx]))]; lam = 0
        q = Cq[i]
        info = dict(rho=float(rho), logacq=float(tot[i]), pof=float(pf[i]))
        if elastic:
            thr = tot[i] - math.log(2.0)
            P = SC.fwd(Cq[tot >= thr])
            centres = []
            for p in P[np.argsort(-tot[tot >= thr])]:
                if all(np.linalg.norm(p - c) > 0.15 for c in centres):
                    centres.append(p)
            info['n_promising'] = len(centres)
        return q, lam, info


# ---------------------------------------------------------------------------
# discrete-event simulation of the execution modes
# ---------------------------------------------------------------------------
def run_campaign(oracle, mode='B', W=4, budget=100, seed=0, prune=True, warm=True,
                 Wmin=2, Wmax=8, sea_target=None, ref=None, cost_ratio=5.0, pool=None,
                 prior_data=None, n0=None, verbose=False, policy='mf'):
    rng = np.random.default_rng(seed)
    pool = pool if pool is not None else load_pool()[0]
    opt = Optimiser(pool, ref, cost_ratio, prior_data=(prior_data if warm else None), rng=rng,
                    policy=policy)
    init_lam = 0 if policy == 'mf' else 1
    from scipy.stats import qmc
    n0 = n0 or W
    init_idx = rng.choice(len(pool), size=n0, replace=False)
    obs = []; events = []; t_master = 0.0
    dispatched = 0; busy = {}; log = []
    trace = []  # (time, n_done, hv)
    occupancy = []  # (worker, start, end, lam, status)
    free_workers = list(range(W if mode != 'C' else Wmax))
    capacity = W if mode != 'C' else W
    pend_lam = {}
    def dispatch(q, lam, t0, worker):
        nonlocal dispatched
        r = oracle.evaluate(q, lam, prune=prune)
        pend_lam[dispatched] = lam
        heapq.heappush(events, (t0 + r['duration'], dispatched, worker, q, lam, r, t0))
        busy[dispatched] = q
        dispatched += 1
    for k in range(n0):
        dispatch(pool[init_idx[k]], init_lam, 0.0, free_workers.pop(0))
    decision_times = []
    hv = 0.0
    W_trace = [(0.0, capacity)]
    while events:
        if mode == 'A':
            # synchronous: wait for whole batch
            batch = []
            tmax = 0.0
            while events:
                ev = heapq.heappop(events); batch.append(ev); tmax = max(tmax, ev[0])
            for ev in batch:
                tf, did, wk, q, lam, r, t0 = ev
                obs.append(dict(q=q, lam=lam, **r, t_end=tf)); busy.pop(did, None)
                occupancy.append((wk, t0, tf, lam, r['status'])); free_workers.append(wk)
            t = tmax
            hv = current_hv(obs, ref); trace.append((t, len(obs), hv))
            if dispatched >= budget:
                break
            nb = min(W, budget - dispatched)
            pend = []
            batch_lams = []
            tdec = 0.0
            for k in range(nb):
                t1 = time.perf_counter()
                opt.update(obs) if k == 0 else None
                opt._pending_hf = sum(1 for k in busy if pend_lam.get(k) == 1) + sum(1 for l in batch_lams if l == 1)
                q, lam, info = opt.propose(obs, pend)
                batch_lams.append(lam)
                tdec += time.perf_counter() - t1
                pend.append(q); log.append(dict(info, lam=lam))
                decision_times.append(time.perf_counter() - t1)
            for q, lam in zip(pend, [l['lam'] for l in log[-nb:]]):
                dispatch(q, lam, t + tdec, free_workers.pop(0))
            continue
        # asynchronous modes (B: fixed pool; C: elastic pool)
        ev = heapq.heappop(events)
        tf, did, wk, q, lam, r, t0 = ev
        obs.append(dict(q=q, lam=lam, **r, t_end=tf)); busy.pop(did, None)
        occupancy.append((wk, t0, tf, lam, r['status'])); free_workers.append(wk)
        t = max(tf, t_master)
        hv = current_hv(obs, ref); trace.append((tf, len(obs), hv))
        if dispatched >= budget:
            continue
        t1 = time.perf_counter()
        opt.update(obs)
        pend = list(busy.values())
        opt._pending_hf = sum(1 for k in busy if pend_lam.get(k) == 1)
        q, lam, info = opt.propose(obs, pend, elastic=(mode == 'C'))
        dt_dec = time.perf_counter() - t1
        decision_times.append(dt_dec); t_master = t + dt_dec
        if mode == 'C':
            capacity = int(np.clip(info.get('n_promising', W), Wmin, Wmax))
            W_trace.append((t_master, capacity))
        if len(busy) < capacity:
            log.append(dict(info, lam=lam, t=t_master, nbusy=len(busy)))
            dispatch(q, lam, t_master, free_workers.pop(0)); pend.append(q)
        while mode == 'C' and len(busy) < capacity and dispatched < budget and free_workers:
            t2 = time.perf_counter()
            opt._pending_hf = sum(1 for k in busy if pend_lam.get(k) == 1)
            q2, lam2, info2 = opt.propose(obs, pend)
            dt2 = time.perf_counter() - t2
            t_master += dt2; decision_times.append(dt2)
            log.append(dict(info2, lam=lam2, t=t_master, nbusy=len(busy)))
            dispatch(q2, lam2, t_master, free_workers.pop(0)); pend.append(q2)
        if not events and dispatched < budget:
            # pool drained by contraction: keep at least one evaluation running
            q2, lam2, info2 = opt.propose(obs, pend)
            dispatch(q2, lam2, t_master, free_workers.pop(0))
    T_wall = max(o['t_end'] for o in obs)
    return summarise(obs, trace, occupancy, decision_times, T_wall, W, mode, ref, sea_target,
                     oracle, W_trace, log)


def current_hv(obs, ref):
    Y = [o['y'] for o in obs if o['lam'] == 1 and o['status'] == 'completed' and o['feasible']]
    return hypervolume(np.array(Y), ref) if Y else 0.0


def summarise(obs, trace, occupancy, decision_times, T_wall, W, mode, ref, sea_target,
              oracle, W_trace, log):
    busy_time = sum(o['duration'] for o in obs)
    pruned = [o for o in obs if o['status'] == 'pruned']
    saved = sum(o['full_duration'] - o['duration'] for o in pruned)
    # practical estimate: median completed solve time at the same fidelity
    med = {lam: np.median([o['duration'] for o in obs if o['status'] == 'completed' and o['lam'] == lam] or [np.nan])
           for lam in (0, 1)}
    saved_est = sum(max(med[o['lam']] - o['duration'], 0.0) for o in pruned if np.isfinite(med[o['lam']]))
    # time-to-target: first HF completed feasible with SEA >= target
    ttt = np.inf
    for o in sorted(obs, key=lambda o: o['t_end']):
        if o['lam'] == 1 and o['status'] == 'completed' and o['feasible'] and o['y'][0] >= sea_target:
            ttt = o['t_end']; break
    if mode == 'C':
        # integral of capacity over time
        tt = [w[0] for w in W_trace] + [T_wall]
        cap = sum((tt[i + 1] - tt[i]) * W_trace[i][1] for i in range(len(W_trace)))
    else:
        cap = W * T_wall
    out = dict(mode=mode, T_wall=T_wall, cpu_hours=busy_time / 3600.0, n_pruned=len(pruned),
               cpu_saved_h=saved / 3600.0, cpu_saved_est_h=saved_est / 3600.0, hv_final=trace[-1][2] if trace else 0.0, ttt=ttt,
               utilisation=busy_time / cap, n_hf=sum(o['lam'] == 1 for o in obs),
               mean_decision_s=float(np.mean(decision_times)) if decision_times else 0.0,
               trace=trace, occupancy=occupancy, W_trace=W_trace,
               obs=[dict(q=list(map(float, o['q'])), lam=int(o['lam']), status=o['status'],
                         y=list(map(float, o['y'])) if o['y'] is not None else None,
                         feasible=bool(o['feasible']), duration=float(o['duration']),
                         full_duration=float(o['full_duration']), t_end=float(o['t_end']),
                         eps_end=float(o.get('eps_end', 0.6)),
                         true_y=list(map(float, o['true_y'])), smax=float(o.get('smax', np.nan))) for o in obs],
               log=log)
    return out
