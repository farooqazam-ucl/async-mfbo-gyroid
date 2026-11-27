"""Explicit dynamic finite-element solver for voxel (structured hexahedral)
meshes of periodic lattice columns under platen compression.

Formulation (updated Lagrangian, central difference):
  * 8-node hexahedra, one-point quadrature, mean-quadrature B at the centroid
  * hypoelastic-plastic J2 material, Jaumann objective rate, radial return,
    Voce + linear isotropic hardening
  * Flanagan-Belytschko stiffness hourglass control (cube-exact bending
    coefficient, tangent-scaled in plasticity) plus light viscous component
  * linear/quadratic bulk viscosity
  * rigid analytical platens (penalty, regularised Coulomb friction)
  * self-contact: pinball penalty contact between surface elements
    (Belytschko & Neal 1991) with friction
  * lateral periodicity with free lateral periods (zero mean lateral stress)
  * selective (conditional) mass scaling to a target time increment

Units: mm, s, tonne, N, MPa.
"""
import json
import os
import time

import numpy as np
from numba import njit

XI = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float64)
ETA = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64)
ZETA = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)
HGV = np.array([[1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
                [-1, 1, -1, 1, 1, -1, 1, -1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# element kernel
# ---------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def element_kernel(x, v, conn, sx, sy, L, Ldot, sig, epsp, Qhg, V0, fac,
                   dt, mat, hgp, bvp, f, FL, cen, cvel, sb2, en, plast, alive, m, me, msp):
    lam = mat[0]; mu = mat[1]; rho = mat[2]; s0 = mat[3]; Qv = mat[4]
    bv = mat[5]; Hl = mat[6]; E = mat[7]
    khg0 = hgp[0]; hvisc = hgp[1]; tfloor = hgp[2]; h0 = hgp[3]
    q1 = bvp[0]; q2 = bvp[1]
    Lx = L[0]; Ly = L[1]; Ldx = Ldot[0]; Ldy = Ldot[1]
    ne = conn.shape[0]
    xe = np.empty((8, 3)); ve = np.empty((8, 3)); b = np.empty((8, 3))
    g = np.empty((4, 8)); fe = np.empty((8, 3))
    ie = 0.0; hge = 0.0; bve = 0.0
    c2 = (lam + 2.0 * mu) / rho
    for e in range(ne):
        cx = 0.0; cy = 0.0; cz = 0.0; vx = 0.0; vy = 0.0; vz = 0.0
        for I in range(8):
            n = conn[e, I]
            xe[I, 0] = x[n, 0] + sx[e, I] * Lx
            xe[I, 1] = x[n, 1] + sy[e, I] * Ly
            xe[I, 2] = x[n, 2]
            ve[I, 0] = v[n, 0] + sx[e, I] * Ldx
            ve[I, 1] = v[n, 1] + sy[e, I] * Ldy
            ve[I, 2] = v[n, 2]
            cx += xe[I, 0]; cy += xe[I, 1]; cz += xe[I, 2]
            vx += ve[I, 0]; vy += ve[I, 1]; vz += ve[I, 2]
        cen[e, 0] = cx * 0.125; cen[e, 1] = cy * 0.125; cen[e, 2] = cz * 0.125
        cvel[e, 0] = vx * 0.125; cvel[e, 1] = vy * 0.125; cvel[e, 2] = vz * 0.125
        if alive[e] == 0:
            continue
        # Jacobian at centroid J_ij = sum_I x_iI dN_I/dxi_j
        J00 = 0.0; J01 = 0.0; J02 = 0.0; J10 = 0.0; J11 = 0.0; J12 = 0.0
        J20 = 0.0; J21 = 0.0; J22 = 0.0
        for I in range(8):
            a0 = XI[I] * 0.125; a1 = ETA[I] * 0.125; a2 = ZETA[I] * 0.125
            J00 += xe[I, 0] * a0; J01 += xe[I, 0] * a1; J02 += xe[I, 0] * a2
            J10 += xe[I, 1] * a0; J11 += xe[I, 1] * a1; J12 += xe[I, 1] * a2
            J20 += xe[I, 2] * a0; J21 += xe[I, 2] * a1; J22 += xe[I, 2] * a2
        C00 = J11 * J22 - J12 * J21; C01 = J12 * J20 - J10 * J22; C02 = J10 * J21 - J11 * J20
        C10 = J02 * J21 - J01 * J22; C11 = J00 * J22 - J02 * J20; C12 = J01 * J20 - J00 * J21
        C20 = J01 * J12 - J02 * J11; C21 = J02 * J10 - J00 * J12; C22 = J00 * J11 - J01 * J10
        det = J00 * C00 + J01 * C01 + J02 * C02
        if not (det > 0.0025 * h0 * h0 * h0):
            # element inverted or compressed below 2% of its volume: erode
            alive[e] = 0
            en[3] += 1.0
            sb2[e] = 0.0
            continue
        idet = 1.0 / det
        # Jinv[j,k] = C[k,j]/det ; b[I,k] = sum_j Jinv[j,k] dN[I,j]
        s2 = 0.0
        for I in range(8):
            a0 = XI[I] * 0.125; a1 = ETA[I] * 0.125; a2 = ZETA[I] * 0.125
            b[I, 0] = (C00 * a0 + C01 * a1 + C02 * a2) * idet
            b[I, 1] = (C10 * a0 + C11 * a1 + C12 * a2) * idet
            b[I, 2] = (C20 * a0 + C21 * a1 + C22 * a2) * idet
            s2 += b[I, 0] * b[I, 0] + b[I, 1] * b[I, 1] + b[I, 2] * b[I, 2]
        sb2[e] = s2
        V = 8.0 * det
        # velocity gradient
        L00 = 0.0; L01 = 0.0; L02 = 0.0; L10 = 0.0; L11 = 0.0; L12 = 0.0
        L20 = 0.0; L21 = 0.0; L22 = 0.0
        for I in range(8):
            L00 += ve[I, 0] * b[I, 0]; L01 += ve[I, 0] * b[I, 1]; L02 += ve[I, 0] * b[I, 2]
            L10 += ve[I, 1] * b[I, 0]; L11 += ve[I, 1] * b[I, 1]; L12 += ve[I, 1] * b[I, 2]
            L20 += ve[I, 2] * b[I, 0]; L21 += ve[I, 2] * b[I, 1]; L22 += ve[I, 2] * b[I, 2]
        D0 = L00; D1 = L11; D2 = L22
        D3 = 0.5 * (L01 + L10); D4 = 0.5 * (L12 + L21); D5 = 0.5 * (L20 + L02)
        W01 = 0.5 * (L01 - L10); W12 = 0.5 * (L12 - L21); W20 = 0.5 * (L20 - L02)
        # stress (Voigt xx yy zz xy yz zx)
        sxx = sig[e, 0]; syy = sig[e, 1]; szz = sig[e, 2]
        sxy = sig[e, 3]; syz = sig[e, 4]; szx = sig[e, 5]
        # Jaumann rotation: dS = W S - S W  (W antisymmetric: W10=-W01 ...)
        # full W matrix
        w01 = W01; w02 = -W20; w12 = W12
        # (WS - SW)_ij computed explicitly
        r00 = 2.0 * (w01 * sxy + w02 * szx)
        r11 = 2.0 * (-w01 * sxy + w12 * syz)
        r22 = 2.0 * (-w02 * szx - w12 * syz)
        # explicit general formula for off-diagonals
        # (WS)_01 = W00 S01 + W01 S11 + W02 S21 = w01*syy + w02*syz
        # (SW)_01 = S00 W01 + S01 W11 + S02 W21 = sxx*w01 + szx*(-w12)
        r01 = (w01 * syy + w02 * syz) - (sxx * w01 - szx * w12)
        # (WS)_12 = W10 S02 + W11 S12 + W12 S22 = -w01*szx + w12*szz
        # (SW)_12 = S10 W02 + S11 W12 + S12 W22 = sxy*w02 + syy*w12
        r12 = (-w01 * szx + w12 * szz) - (sxy * w02 + syy * w12)
        # (WS)_20 = W20 S00 + W21 S10 + W22 S20 = -w02*sxx - w12*sxy
        # (SW)_20 = S20 W00 + S21 W10 + S22 W20 = syz*(-w01) + szz*(-w02)
        r20 = (-w02 * sxx - w12 * sxy) - (-syz * w01 - szz * w02)
        trD = D0 + D1 + D2
        sxx += dt * (r00 + lam * trD + 2.0 * mu * D0)
        syy += dt * (r11 + lam * trD + 2.0 * mu * D1)
        szz += dt * (r22 + lam * trD + 2.0 * mu * D2)
        sxy += dt * (r01 + 2.0 * mu * D3)
        syz += dt * (r12 + 2.0 * mu * D4)
        szx += dt * (r20 + 2.0 * mu * D5)
        # radial return
        p = (sxx + syy + szz) / 3.0
        dxx = sxx - p; dyy = syy - p; dzz = szz - p
        J2 = 0.5 * (dxx * dxx + dyy * dyy + dzz * dzz) + sxy * sxy + syz * syz + szx * szx
        qtr = np.sqrt(3.0 * J2)
        ep = epsp[e]
        sy_ = s0 + Qv * (1.0 - np.exp(-bv * ep)) + Hl * ep
        tscale = 1.0
        plast[e] = 0
        if qtr > sy_:
            dg = 0.0
            for it in range(8):
                ex = np.exp(-bv * (ep + dg))
                syv = s0 + Qv * (1.0 - ex) + Hl * (ep + dg)
                Hv = Qv * bv * ex + Hl
                gfun = qtr - 3.0 * mu * dg - syv
                dg += gfun / (3.0 * mu + Hv)
                if abs(gfun) < 1e-9 * s0:
                    break
            ratio = 1.0 - 3.0 * mu * dg / qtr
            dxx *= ratio; dyy *= ratio; dzz *= ratio
            sxy *= ratio; syz *= ratio; szx *= ratio
            sxx = dxx + p; syy = dyy + p; szz = dzz + p
            epsp[e] = ep + dg
            Hv = Qv * bv * np.exp(-bv * (ep + dg)) + Hl
            tscale = Hv / (E + Hv)
            if tscale < tfloor:
                tscale = tfloor
            plast[e] = 1
        sig[e, 0] = sxx; sig[e, 1] = syy; sig[e, 2] = szz
        sig[e, 3] = sxy; sig[e, 4] = syz; sig[e, 5] = szx
        if mat[8] > 0.0 and epsp[e] > mat[8]:
            alive[e] = 0
            en[3] += 1.0
            continue
        # internal energy (stress power)
        ie += dt * V * (sxx * D0 + syy * D1 + szz * D2 + 2.0 * (sxy * D3 + syz * D4 + szx * D5))
        # bulk viscosity
        rhoe = rho * fac[e] * V0[e] / V
        qb = 0.0
        if trD < 0.0:
            le = V ** (1.0 / 3.0)
            ce = np.sqrt(c2 / fac[e])
            qb = rhoe * le * (q1 * ce * (-trD) + q2 * q2 * le * trD * trD)
            bve += dt * V * qb * (-trD)
        # internal forces (residual) f_I = -V (sigma - qb I) b_I
        for I in range(8):
            fe[I, 0] = -V * ((sxx - qb) * b[I, 0] + sxy * b[I, 1] + szx * b[I, 2])
            fe[I, 1] = -V * (sxy * b[I, 0] + (syy - qb) * b[I, 1] + syz * b[I, 2])
            fe[I, 2] = -V * (szx * b[I, 0] + syz * b[I, 1] + (szz - qb) * b[I, 2])
        # hourglass control
        khg = khg0 * tscale
        cvh = hvisc * rhoe * np.sqrt(c2 / fac[e]) * h0 * h0
        for a in range(4):
            hx = 0.0; hy = 0.0; hz = 0.0
            for J in range(8):
                hx += HGV[a, J] * xe[J, 0]; hy += HGV[a, J] * xe[J, 1]; hz += HGV[a, J] * xe[J, 2]
            for I in range(8):
                g[a, I] = HGV[a, I] - (hx * b[I, 0] + hy * b[I, 1] + hz * b[I, 2])
        # on-the-fly conditional mass scaling: element frequency bound including
        # the hourglass stiffness; add mass to the element nodes if needed
        if msp[0] > 0.0:
            gmax = 0.0
            for a in range(4):
                gg = 0.0
                for I in range(8):
                    gg += g[a, I] * g[a, I]
                if gg > gmax:
                    gmax = gg
            w2 = 8.0 * (lam + 2.0 * mu) * s2 + 8.0 * khg0 * gmax / V0[e]
            need = w2 / rho * msp[0]
            if need > fac[e]:
                dm = rho * V0[e] * (need - fac[e])
                for I in range(8):
                    m[conn[e, I]] += 0.125 * dm
                fac[e] = need
                me[e] = rho * need * V0[e]
                en[4] += dm
        for a in range(4):
            qx = 0.0; qy = 0.0; qz = 0.0
            for I in range(8):
                qx += g[a, I] * ve[I, 0]; qy += g[a, I] * ve[I, 1]; qz += g[a, I] * ve[I, 2]
            Qx = Qhg[e, a, 0]; Qy = Qhg[e, a, 1]; Qz = Qhg[e, a, 2]
            # rotate with spin: dQ = W Q
            rQx = w01 * Qy + w02 * Qz
            rQy = -w01 * Qx + w12 * Qz
            rQz = -w02 * Qx - w12 * Qy
            Qx += dt * (rQx + khg * qx)
            Qy += dt * (rQy + khg * qy)
            Qz += dt * (rQz + khg * qz)
            Qhg[e, a, 0] = Qx; Qhg[e, a, 1] = Qy; Qhg[e, a, 2] = Qz
            hge += dt * (Qx * qx + Qy * qy + Qz * qz)
            Fx = Qx + cvh * qx; Fy = Qy + cvh * qy; Fz = Qz + cvh * qz
            for I in range(8):
                fe[I, 0] -= Fx * g[a, I]
                fe[I, 1] -= Fy * g[a, I]
                fe[I, 2] -= Fz * g[a, I]
            hge += dt * cvh * (qx * qx + qy * qy + qz * qz)
        # scatter
        for I in range(8):
            n = conn[e, I]
            f[n, 0] += fe[I, 0]; f[n, 1] += fe[I, 1]; f[n, 2] += fe[I, 2]
            if sx[e, I]:
                FL[0] += fe[I, 0]
            if sy[e, I]:
                FL[1] += fe[I, 1]
    en[0] += ie; en[1] += hge; en[2] += bve


# ---------------------------------------------------------------------------
# platen contact
# ---------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def platen_kernel(x, v, m, f, zbot, ztop, vtop, dt, kfac, zeta, mu, out):
    nn = x.shape[0]
    Ft = 0.0; Fb = 0.0; wnode = 0.0
    for n in range(nn):
        z = x[n, 2]
        pen = 0.0; rate = 0.0; sgn = 0.0
        if z < zbot:
            pen = zbot - z; rate = -v[n, 2]; sgn = 1.0
        elif z > ztop:
            pen = z - ztop; rate = v[n, 2] - vtop; sgn = -1.0
        if pen > 0.0:
            mn = m[n]
            k = kfac * mn / (dt * dt)
            c = 2.0 * zeta * np.sqrt(k * mn)
            fn = k * pen + c * rate
            if fn < 0.0:
                fn = 0.0
            fz = sgn * fn
            f[n, 2] += fz
            # friction (platen has no lateral velocity)
            vtx = v[n, 0]; vty = v[n, 1]
            vt = np.sqrt(vtx * vtx + vty * vty)
            fxn = 0.0; fyn = 0.0
            if vt > 0.0 and mu > 0.0:
                fmax = mu * fn
                fstick = 0.5 * mn * vt / dt
                ft = fmax if fmax < fstick else fstick
                fxn = -ft * vtx / vt; fyn = -ft * vty / vt
                f[n, 0] += fxn; f[n, 1] += fyn
            wnode += dt * (fz * v[n, 2] + fxn * v[n, 0] + fyn * v[n, 1])
            if sgn > 0:
                Fb += fn
            else:
                Ft += fn
    out[0] = Ft; out[1] = Fb; out[2] = wnode


# ---------------------------------------------------------------------------
# self contact (pinball)
# ---------------------------------------------------------------------------
@njit(cache=True)
def build_pairs(sidx, cen, eijk, N, L, rc, maxpairs):
    ns = sidx.size
    Lx = L[0]; Ly = L[1]
    ncx = max(1, int(Lx / rc)); ncy = max(1, int(Ly / rc))
    zmin = 1e30; zmax = -1e30
    for s in range(ns):
        z = cen[sidx[s], 2]
        if not np.isfinite(z):
            continue
        if z < zmin: zmin = z
        if z > zmax: zmax = z
    ncz = max(1, int((zmax - zmin) / rc) + 1)
    csx = Lx / ncx; csy = Ly / ncy
    cid = np.empty(ns, dtype=np.int64)
    for s in range(ns):
        e = sidx[s]
        if not (np.isfinite(cen[e, 0]) and np.isfinite(cen[e, 1]) and np.isfinite(cen[e, 2])):
            cid[s] = 0
            continue
        xx = cen[e, 0] % Lx; yy = cen[e, 1] % Ly
        ix = int(xx / csx) % ncx; iy = int(yy / csy) % ncy
        iz = int((cen[e, 2] - zmin) / rc)
        if iz >= ncz: iz = ncz - 1
        if iz < 0: iz = 0
        cid[s] = (iz * ncy + iy) * ncx + ix
    ncell = ncx * ncy * ncz
    start = np.zeros(ncell + 1, dtype=np.int64)
    for s in range(ns):
        start[cid[s] + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    order = np.empty(ns, dtype=np.int64)
    fill = start[:-1].copy()
    for s in range(ns):
        order[fill[cid[s]]] = s; fill[cid[s]] += 1
    pairs = np.empty((maxpairs, 2), dtype=np.int64)
    npair = 0
    rc2 = rc * rc
    for s in range(ns):
        e = sidx[s]
        c = cid[s]
        ix = c % ncx; iy = (c // ncx) % ncy; iz = c // (ncx * ncy)
        for dz in range(-1, 2):
            jz = iz + dz
            if jz < 0 or jz >= ncz: continue
            for dy in range(-1, 2):
                jy = (iy + dy) % ncy
                if ncy < 3 and dy != 0 and (iy + dy) % ncy == iy: continue
                for dx in range(-1, 2):
                    jx = (ix + dx) % ncx
                    if ncx < 3 and dx != 0 and (ix + dx) % ncx == ix: continue
                    cc = (jz * ncy + jy) * ncx + jx
                    for kk in range(start[cc], start[cc + 1]):
                        t = order[kk]
                        if t <= s: continue
                        e2 = sidx[t]
                        di = abs(eijk[e, 0] - eijk[e2, 0]); di = min(di, N - di)
                        dj = abs(eijk[e, 1] - eijk[e2, 1]); dj = min(dj, N - dj)
                        dk = abs(eijk[e, 2] - eijk[e2, 2])
                        if di <= 1 and dj <= 1 and dk <= 1: continue
                        ddx = cen[e2, 0] - cen[e, 0]; ddx -= Lx * np.round(ddx / Lx)
                        ddy = cen[e2, 1] - cen[e, 1]; ddy -= Ly * np.round(ddy / Ly)
                        ddz = cen[e2, 2] - cen[e, 2]
                        if ddx * ddx + ddy * ddy + ddz * ddz < rc2:
                            if npair < maxpairs:
                                pairs[npair, 0] = e; pairs[npair, 1] = e2
                            npair += 1
    return pairs, npair


@njit(fastmath=True, cache=True)
def pinball_kernel(pairs, npair, cen, cvel, conn, sx, sy, me, f, FL, L, r2, dt, kfac,
                   zeta, mu, out, alive):
    Lx = L[0]; Ly = L[1]
    wsum = 0.0; nact = 0
    for p in range(npair):
        a = pairs[p, 0]; b = pairs[p, 1]
        if alive[a] == 0 or alive[b] == 0:
            continue
        dx = cen[b, 0] - cen[a, 0]; nx_ = np.round(dx / Lx); dx -= Lx * nx_
        dy = cen[b, 1] - cen[a, 1]; ny_ = np.round(dy / Ly); dy -= Ly * ny_
        dz = cen[b, 2] - cen[a, 2]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 >= r2 * r2:
            continue
        d = np.sqrt(d2)
        if d < 1e-12:
            continue
        nx = dx / d; ny = dy / d; nz = dz / d
        pen = r2 - d
        mm = me[a] if me[a] < me[b] else me[b]
        k = kfac * mm / (dt * dt)
        c = 2.0 * zeta * np.sqrt(k * mm * 0.5)
        rvx = cvel[b, 0] - cvel[a, 0]; rvy = cvel[b, 1] - cvel[a, 1]; rvz = cvel[b, 2] - cvel[a, 2]
        vn = rvx * nx + rvy * ny + rvz * nz
        fn = k * pen - c * vn
        if fn < 0.0:
            fn = 0.0
        # force on b along +n, on a along -n
        fbx = fn * nx; fby = fn * ny; fbz = fn * nz
        tx = rvx - vn * nx; ty = rvy - vn * ny; tz = rvz - vn * nz
        vt = np.sqrt(tx * tx + ty * ty + tz * tz)
        if vt > 0.0 and mu > 0.0:
            fmax = mu * fn
            fstick = 0.25 * mm * vt / dt
            ft = fmax if fmax < fstick else fstick
            fbx -= ft * tx / vt; fby -= ft * ty / vt; fbz -= ft * tz / vt
        for I in range(8):
            nb = conn[b, I]; na = conn[a, I]
            f[nb, 0] += 0.125 * fbx; f[nb, 1] += 0.125 * fby; f[nb, 2] += 0.125 * fbz
            f[na, 0] -= 0.125 * fbx; f[na, 1] -= 0.125 * fby; f[na, 2] -= 0.125 * fbz
            FL[0] += 0.125 * fbx * (sx[b, I] - sx[a, I])
            FL[1] += 0.125 * fby * (sy[b, I] - sy[a, I])
        # periodic box generalised force: b was shifted by -n*L
        FL[0] += fbx * (-nx_); FL[1] += fby * (-ny_)
        wsum += dt * (fbx * rvx + fby * rvy + fbz * rvz)
        nact += 1
    out[0] = wsum; out[1] = nact


@njit(fastmath=True, cache=True)
def integrate(x, v, m, f, dt):
    nn = x.shape[0]
    ke = 0.0
    for n in range(nn):
        im = dt / m[n]
        v[n, 0] += f[n, 0] * im; v[n, 1] += f[n, 1] * im; v[n, 2] += f[n, 2] * im
        x[n, 0] += dt * v[n, 0]; x[n, 1] += dt * v[n, 1]; x[n, 2] += dt * v[n, 2]
        ke += 0.5 * m[n] * (v[n, 0] ** 2 + v[n, 1] ** 2 + v[n, 2] ** 2)
        f[n, 0] = 0.0; f[n, 1] = 0.0; f[n, 2] = 0.0
    return ke


@njit(cache=True)
def nodal_mass(conn, V0, fac, rho, nn):
    m = np.zeros(nn)
    for e in range(conn.shape[0]):
        me = rho * fac[e] * V0[e] * 0.125
        for I in range(8):
            m[conn[e, I]] += me
    return m


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
DEFAULT_MAT = dict(E=1650.0, nu=0.40, rho=9.30e-10, sig0=30.0, Qv=20.0, bv=30.0, Hl=0.0)


class Simulation:
    def __init__(self, mesh, mat=None, strain_max=0.6, T=None, T_factor=20.0,
                 mu_platen=0.2, mu_self=0.2, dt_safety=0.75, ramp_frac=0.04,
                 hg_coef=1.0 / 48.0, hg_visc=0.02, hg_floor=0.02,
                 kpen=0.5, kpin=0.15, barostat=True, hist_dstrain=0.001,
                 sigma_limit=None, prune_window=5, prune_skip=0.01,
                 stop_file=None, snap_strains=(), verbose=False):
        self.mesh = mesh
        mat = dict(DEFAULT_MAT, **(mat or {}))
        self.matd = mat
        E, nu, rho = mat["E"], mat["nu"], mat["rho"]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
        self.mat = np.array([lam, mu, rho, mat["sig0"], mat["Qv"], mat["bv"], mat["Hl"], E,
                             mat.get("eps_fail", 0.0)])
        self.cd = np.sqrt((lam + 2 * mu) / rho)
        h0 = mesh["h"]
        self.h0 = h0
        Eb = E / (1 - nu * nu)
        self.hgp = np.array([hg_coef * Eb * h0, hg_visc, hg_floor, h0])
        self.bvp = np.array([0.06, 1.2])
        self.x = mesh["nodes"].copy()
        self.v = np.zeros_like(self.x)
        self.f = np.zeros_like(self.x)
        self.conn = mesh["elems"]
        self.sx = mesh["sx"]; self.sy = mesh["sy"]
        ne = self.conn.shape[0]; nn = self.x.shape[0]
        self.ne, self.nn = ne, nn
        self.sig = np.zeros((ne, 6)); self.epsp = np.zeros(ne)
        self.Qhg = np.zeros((ne, 4, 3)); self.V0 = np.full(ne, h0 ** 3)
        self.fac = np.ones(ne); self.plast = np.zeros(ne, dtype=np.int8)
        self.alive = np.ones(ne, dtype=np.int8)
        self.cen = np.zeros((ne, 3)); self.cvel = np.zeros((ne, 3)); self.sb2 = np.zeros(ne)
        self.m = nodal_mass(self.conn, self.V0, self.fac, rho, nn)
        self.m0total = self.m.sum()
        self.L = np.array([mesh["Lx"], mesh["Ly"]], dtype=np.float64)
        self.L0 = self.L.copy()
        self.Ldot = np.zeros(2); self.FL = np.zeros(2)
        self.H = mesh["H"]; self.A0 = self.L0[0] * self.L0[1]
        self.barostat = barostat
        # time step from cube bound omega^2 = 12 c^2/h^2 (FB), safety
        self.dt = dt_safety * 2.0 / np.sqrt(12.0) * h0 / self.cd
        self.dt_target = self.dt
        # loading time
        rel = mesh["rho"]
        Estar = 892.0 * rel ** 1.39  # MPa, Abueidda et al. 2019 fit (estimate only)
        cstar = np.sqrt(Estar / (rel * rho))
        self.T1 = 2.0 * self.H / cstar
        self.T = T if T is not None else T_factor * self.T1 * (strain_max / 0.6)
        self.tr = ramp_frac * self.T
        self.delta_max = strain_max * self.H
        self.Vp = self.delta_max / (self.T - 0.5 * self.tr)
        self.strain_max = strain_max
        # barostat
        Mtot = self.m0total
        self.ML = Mtot
        kL = Estar * self.H
        self.cL = 2.0 * 0.7 * np.sqrt(kL * self.ML)
        # contact
        self.kpen = kpen; self.kpin = kpin
        self.mu_platen = mu_platen; self.mu_self = mu_self
        self.sidx = np.nonzero(mesh["surf"])[0].astype(np.int64)
        self.eijk = mesh["eijk"]
        self.rpin2 = h0  # sum of radii (r = h0/2)
        self.skin = 0.6 * h0
        self.maxpairs = max(200000, 60 * self.sidx.size)
        self.me = rho * self.fac * self.V0
        # (dt/(2*0.9))^2 factor for the per-element stability check
        self.msp = np.array([(self.dt / (2.0 * 0.9)) ** 2])
        self.en = np.zeros(5)
        self.pout = np.zeros(3); self.cout = np.zeros(2)
        self.hist_dstrain = hist_dstrain
        self.sigma_limit = sigma_limit; self.prune_window = prune_window
        self.prune_skip = prune_skip; self.stop_file = stop_file
        self.snap_strains = sorted(snap_strains)
        self.verbose = verbose

    def delta(self, t):
        tr, V = self.tr, self.Vp
        if t < tr:
            tau = t / tr
            return V * tr * (2.5 * tau ** 4 - 3 * tau ** 5 + tau ** 6), V * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
        return V * tr * 0.5 + V * (t - tr), V

    def update_mass_scaling(self):
        lam, mu, rho = self.mat[0], self.mat[1], self.mat[2]
        # omega_e^2 = 8 (lam+2mu)/(rho fac) * sum b^2 ; need 2/omega >= dt/0.9
        need = 8.0 * (lam + 2 * mu) / rho * self.sb2 * (self.dt / (2.0 * 0.9)) ** 2
        newfac = np.maximum(self.fac, need)
        if np.any(newfac > self.fac * 1.0001):
            self.fac = newfac
            self.m = nodal_mass(self.conn, self.V0, self.fac, rho, self.nn)
            self.me = rho * self.fac * self.V0
        # (dt/(2*0.9))^2 factor for the per-element stability check
        self.msp = np.array([(self.dt / (2.0 * 0.9)) ** 2])

    def run(self, out_prefix=None):
        t = 0.0; step = 0
        hist = {k: [] for k in ("strain", "stress", "stress_bot", "KE", "IE", "HG", "BV",
                                "Eself", "Wplat", "Lx", "Ly", "wall", "added_mass", "npair_act", "ninv")}
        ninc = max(1, int(round(self.hist_dstrain * self.H / (self.Vp * self.dt))))
        t0 = time.perf_counter()
        pairs = np.zeros((1, 2), dtype=np.int64); npair = 0
        last_build_cen = None
        Eself = 0.0; Wplat = 0.0; Wext = 0.0
        status = "completed"; ke = 0.0
        snaps = {}; snap_i = 0
        # initial element pass to get centroids
        element_kernel(self.x, self.v, self.conn, self.sx, self.sy, self.L, self.Ldot,
                       self.sig, self.epsp, self.Qhg, self.V0, self.fac, 0.0, self.mat,
                       self.hgp, self.bvp, self.f, self.FL, self.cen, self.cvel, self.sb2,
                       self.en, self.plast, self.alive, self.m, self.me, self.msp)
        self.f[:] = 0.0; self.FL[:] = 0.0
        sF = 0.0
        while True:
            d, vtop = self.delta(t)
            strain = d / self.H
            # contact list maintenance
            if last_build_cen is None or step % 10 == 0:
                rebuild = last_build_cen is None
                if not rebuild:
                    dd = self.cen[self.sidx] - last_build_cen
                    dd[:, 0] -= self.L[0] * np.round(dd[:, 0] / self.L[0])
                    dd[:, 1] -= self.L[1] * np.round(dd[:, 1] / self.L[1])
                    rebuild = dd.size > 0 and np.max(np.abs(dd)) > 0.45 * self.skin
                if rebuild:
                    pairs, npair = build_pairs(self.sidx, self.cen, self.eijk, self.mesh["N"],
                                               self.L, self.rpin2 + self.skin, self.maxpairs)
                    if npair > 200 * max(self.sidx.size, 1):
                        status = "failed"; break
                    if npair > self.maxpairs:
                        self.maxpairs = int(npair * 1.5)
                        pairs, npair = build_pairs(self.sidx, self.cen, self.eijk, self.mesh["N"],
                                                   self.L, self.rpin2 + self.skin, self.maxpairs)
                    last_build_cen = self.cen[self.sidx].copy()
            # forces at current configuration
            ztop = self.H - d
            pinball_kernel(pairs, npair, self.cen, self.cvel, self.conn, self.sx, self.sy, self.me, self.f, self.FL,
                           self.L, self.rpin2, self.dt, self.kpin, 0.1, self.mu_self, self.cout, self.alive)
            platen_kernel(self.x, self.v, self.m, self.f, 0.0, ztop, -vtop, self.dt, self.kpen,
                          0.2, self.mu_platen, self.pout)
            Eself -= self.cout[0]
            Wplat += self.pout[2]
            Ftop = self.pout[0]; Fbot = self.pout[1]
            Wext += Ftop * vtop * self.dt
            # history
            if step % ninc == 0:
                hist["strain"].append(strain); hist["stress"].append(Ftop / self.A0)
                hist["stress_bot"].append(Fbot / self.A0)
                hist["KE"].append(ke); hist["IE"].append(self.en[0]); hist["HG"].append(self.en[1])
                hist["BV"].append(self.en[2]); hist["Eself"].append(Eself); hist["Wplat"].append(Wplat)
                hist["Lx"].append(self.L[0]); hist["Ly"].append(self.L[1])
                hist["wall"].append(time.perf_counter() - t0)
                hist["added_mass"].append(self.m.sum() / self.m0total - 1.0)
                hist["npair_act"].append(self.cout[1]); hist["ninv"].append(float(self.ne - self.alive.sum()))
                while snap_i < len(self.snap_strains) and strain >= self.snap_strains[snap_i] - 1e-9:
                    snaps["%.3f" % self.snap_strains[snap_i]] = self._snapshot()
                    snap_i += 1
                if self.verbose and len(hist["strain"]) % 50 == 1:
                    print("eps=%.3f sig=%.3f KE/IE=%.3f HG/IE=%.3f pairs=%d act=%d added=%.4f wall=%.1f" % (
                        strain, Ftop / self.A0, ke / max(self.en[0], 1e-30), self.en[1] / max(self.en[0], 1e-30),
                        npair, self.cout[1], hist["added_mass"][-1], hist["wall"][-1]), flush=True)
                if not np.isfinite(Ftop) or not np.isfinite(ke):
                    status = "failed"; break
                # in-solver pruning (moving average)
                if self.sigma_limit is not None and strain > self.prune_skip:
                    w = self.prune_window
                    if len(hist["stress"]) >= w:
                        if np.mean(hist["stress"][-w:]) > self.sigma_limit:
                            status = "pruned"; break
                if self.stop_file is not None and os.path.exists(self.stop_file):
                    status = "terminated"; break
            if strain >= self.strain_max - 1e-12:
                break
            # advance: velocities/positions
            ke = integrate(self.x, self.v, self.m, self.f, self.dt)
            if self.barostat:
                self.Ldot += self.dt * (self.FL - self.cL * self.Ldot) / self.ML
                self.L += self.dt * self.Ldot
            self.FL[:] = 0.0
            t += self.dt; step += 1
            element_kernel(self.x, self.v, self.conn, self.sx, self.sy, self.L, self.Ldot,
                           self.sig, self.epsp, self.Qhg, self.V0, self.fac, self.dt, self.mat,
                           self.hgp, self.bvp, self.f, self.FL, self.cen, self.cvel, self.sb2,
                           self.en, self.plast, self.alive, self.m, self.me, self.msp)
        wall = time.perf_counter() - t0
        while snap_i < len(self.snap_strains):
            snaps["%.3f" % self.snap_strains[snap_i]] = self._snapshot(); snap_i += 1
        res = {k: np.asarray(vv) for k, vv in hist.items()}
        res["wall_hist"] = res.pop("wall")
        res.update(status=status, wall=wall, steps=step, dt=self.dt, T=self.T, T1=self.T1,
                   ne=self.ne, nn=self.nn, rho=float(self.mesh["rho"]), h=self.h0,
                   s=float(self.mesh.get("s", np.nan)), Wext=Wext, strain_end=strain)
        res["snaps"] = snaps
        if out_prefix is not None:
            save_result(res, out_prefix, self)
        return res

    def _snapshot(self):
        return dict(x=self.x.astype(np.float32).copy(), L=self.L.copy(),
                    epsp=self.epsp.astype(np.float32).copy())


def save_result(res, prefix, sim=None):
    arrs = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    meta = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in res.items() if not isinstance(v, (np.ndarray, dict))}
    np.savez_compressed(prefix + "_hist.npz", **arrs)
    if res.get("snaps"):
        sd = {}
        for key, s in res["snaps"].items():
            sd["x_" + key] = s["x"]; sd["L_" + key] = s["L"]; sd["ep_" + key] = s["epsp"]
        if sim is not None:
            sd["conn"] = sim.conn; sd["sx"] = sim.sx; sd["sy"] = sim.sy
            sd["eijk"] = sim.eijk
        np.savez_compressed(prefix + "_snaps.npz", **sd)
    with open(prefix + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
