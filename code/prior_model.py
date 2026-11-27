"""Analytical (Gibson-Ashby, layers-in-series) model of a thickness-graded
sheet-gyroid column, used as a zero-cost information source (prior task).

Each axial slice at height u has local relative density rho_loc(t(u))
(tabulated for a uniform sheet).  Using the power laws fitted to PA2200 sheet
gyroids by Abueidda et al. (2019): E* = 892 rho^1.39 MPa, sigma* = 54 rho^1.63
MPa, and a Gibson-Ashby densification strain eps_D = 1 - 1.4 rho, slices in
series collapse in order of increasing strength, each contributing a strain
eps_D * du at a constant stress equal to its strength.
"""
import numpy as np
from geometry import ContinuumDensity, bezier_profile
from metrics import curve_metrics

_CD = None


def cd():
    global _CD
    if _CD is None:
        _CD = ContinuumDensity()
    return _CD


def prior_curve(q, rho_target=0.25, nslice=200, emax=0.6, C2=54.0, n2=1.63,
                C1=892.0, n1=1.39, alpha=1.4):
    c = cd()
    s = c.scale(q, rho_target)
    u = (np.arange(nslice) + 0.5) / nslice
    t = s * bezier_profile(u, q)
    rl = np.interp(t, c.t, c.rho_loc)
    sig = C2 * rl ** n2
    Ei = C1 * rl ** n1
    eD = np.clip(1.0 - alpha * rl, 0.05, 0.95)
    order = np.argsort(sig)
    du = 1.0 / nslice
    S = sig[order]; Eo = Ei[order]; eDo = eD[order]
    A = np.concatenate([[0.0], np.cumsum(du * eDo)[:-1]])          # collapsed slices
    Binv = np.cumsum((du / Eo)[::-1])[::-1]                          # elastic slices j>=k
    e_start = A + S * Binv
    e_end = e_start + du * (eDo - S / Eo)
    e_pts = np.concatenate([[0.0], np.ravel(np.column_stack([e_start, e_end]))])
    s_pts = np.concatenate([[0.0], np.ravel(np.column_stack([S, S]))])
    e_pts = np.maximum.accumulate(e_pts)
    # densification beyond full collapse: steep rise
    e_pts = np.append(e_pts, e_pts[-1] + 0.05); s_pts = np.append(s_pts, s_pts[-1] * 4)
    eg = np.linspace(0, emax, 601)
    sg = np.interp(eg, e_pts, s_pts)
    return eg, sg


def prior_objectives(q, rho_target=0.25):
    e, s = prior_curve(q, rho_target)
    m = curve_metrics(e, s, rho_eff=rho_target * 930.0)
    m['ip'] = float(np.max(s[e <= 0.1 + 1e-12]))
    m['smax'] = float(np.max(s))
    return m
