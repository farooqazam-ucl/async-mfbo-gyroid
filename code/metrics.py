"""Performance metrics of a nominal stress-strain response (Section 4)."""
import numpy as np

EPS_A, EPS_B = 0.10, 0.50       # plateau window
EPS_MIN = 0.05                  # lower bound of densification search


def cumulative_energy(e, s):
    return np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(e))])


def curve_metrics(e, s, rho_eff, eps_min=EPS_MIN, ea=EPS_A, eb=EPS_B, smooth=5):
    """e, s: nominal strain and stress (MPa).  rho_eff: effective density of the
    specimen in kg/m^3 (relative density x solid density).

    Returns densification strain (efficiency maximum), W_v (MJ/m^3), SEA
    (kJ/kg), peak stress up to eps_d (MPa), plateau mean (MPa) and plateau
    variance (MPa^2) over [ea, eb] on a uniform strain grid."""
    e = np.asarray(e, float); s = np.asarray(s, float)
    if smooth and smooth > 1 and s.size > smooth:
        k = np.ones(smooth) / smooth
        ss = np.convolve(s, k, mode='same')
        ss[: smooth // 2] = s[: smooth // 2]; ss[-(smooth // 2):] = s[-(smooth // 2):]
    else:
        ss = s
    W = cumulative_energy(e, s)
    eff = W / np.maximum(ss, 1e-9)
    m = (e >= eps_min)
    idx = np.nonzero(m)[0][np.argmax(eff[m])]
    ed = float(e[idx]); Wd = float(W[idx])
    peak = float(np.max(s[: idx + 1]))
    g = np.linspace(ea, eb, 401)
    sg = np.interp(g, e, s)
    pm = float(sg.mean()); pv = float(sg.var())
    sea_ed = 1000.0 * Wd / rho_eff  # MJ/m3 / (kg/m3) * 1000 = kJ/kg
    sea = 1000.0 * float(W[-1]) / rho_eff   # energy over the full stroke
    return dict(eps_d=ed, Wv=Wd, SEA=sea, SEA_ed=sea_ed, peak=peak, plat_mean=pm, plat_var=pv,
                W60=float(W[-1]))


def first_crossing(e, s, limit, window=5, skip=0.01):
    """Strain at which the moving average of the last `window` samples first
    exceeds `limit` (the in-solver pruning rule); np.inf if never."""
    s = np.asarray(s); e = np.asarray(e)
    for i in range(window - 1, len(s)):
        if e[i] > skip and s[i - window + 1: i + 1].mean() > limit:
            return float(e[i])
    return np.inf
