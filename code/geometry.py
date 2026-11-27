"""Sheet-gyroid geometry, Bezier thickness grading, density normalisation and
implicit voxel meshing with lateral periodicity.

Units: mm.  The specimen column is [0,a] x [0,a] x [0,nz*a] with periodic
lateral faces; it represents the interior of a laterally extended lattice.
"""
import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------
# Gyroid field and gradient-normalised approximate signed distance
# --------------------------------------------------------------------------

def gyroid(X, Y, Z):
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def gyroid_phase_grad(X, Y, Z):
    gx = np.cos(X) * np.cos(Y) - np.sin(Z) * np.sin(X)
    gy = -np.sin(X) * np.sin(Y) + np.cos(Y) * np.cos(Z)
    gz = -np.sin(Y) * np.sin(Z) + np.cos(Z) * np.cos(X)
    return gx, gy, gz


def d_app(x, y, z, a):
    """First-order signed distance G/|grad_r G| in mm (Eq. approx_distance)."""
    k = 2.0 * np.pi / a
    X, Y, Z = k * x, k * y, k * z
    G = gyroid(X, Y, Z)
    gx, gy, gz = gyroid_phase_grad(X, Y, Z)
    gn = k * np.sqrt(gx * gx + gy * gy + gz * gz)
    return G / np.maximum(gn, 1e-12)


# --------------------------------------------------------------------------
# Cubic Bezier thickness law
# --------------------------------------------------------------------------

def bernstein3(u):
    u = np.asarray(u, dtype=float)
    return np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3], axis=-1)


def bezier_profile(u, q):
    """Unscaled profile b(u;q) = sum_i B_i(u) q_i."""
    return bernstein3(u) @ np.asarray(q, dtype=float)


def bezier_min(q):
    """Exact minimum of the cubic Bezier profile on [0,1] (unscaled)."""
    p0, p1, p2, p3 = [float(v) for v in q]
    # t'(u)/3 = A(1-u)^2 + 2B u(1-u) + C u^2 with A=p1-p0, B=p2-p1, C=p3-p2
    A, B, C = p1 - p0, p2 - p1, p3 - p2
    # expand: (A - 2B + C) u^2 + 2(B - A) u + A
    c2, c1, c0 = A - 2 * B + C, 2 * (B - A), A
    cands = [0.0, 1.0]
    if abs(c2) > 1e-14:
        disc = c1 * c1 - 4 * c2 * c0
        if disc >= 0:
            r = np.sqrt(disc)
            for uu in ((-c1 + r) / (2 * c2), (-c1 - r) / (2 * c2)):
                if 0.0 < uu < 1.0:
                    cands.append(uu)
    elif abs(c1) > 1e-14:
        uu = -c0 / c1
        if 0.0 < uu < 1.0:
            cands.append(uu)
    vals = bezier_profile(np.array(cands), q)
    return float(np.min(vals))


# --------------------------------------------------------------------------
# Voxel grid of approximate distances
# --------------------------------------------------------------------------

class VoxelGrid:
    """Cell-centred |d_app| on an N x N x (nz*N) grid for cell size a."""

    def __init__(self, a=25.0, N=36, nz=4):
        self.a, self.N, self.nz = float(a), int(N), int(nz)
        self.Nz = self.nz * self.N
        self.h = self.a / self.N
        self.H = self.a * self.nz
        c = (np.arange(self.N) + 0.5) * self.h
        cz = (np.arange(self.Nz) + 0.5) * self.h
        X, Y, Z = np.meshgrid(c, c, cz, indexing="ij")
        self.absd = np.abs(d_app(X, Y, Z, self.a)).astype(np.float64)
        self.u = cz / self.H
        # sorted |d| per z-layer for fast counting
        self.sorted_layers = [np.sort(self.absd[:, :, k].ravel()) for k in range(self.Nz)]
        self.nvox = self.N * self.N * self.Nz

    def fraction(self, s, b):
        """Solid fraction for thickness t_k = s*b_k (b_k per layer)."""
        tot = 0
        for k in range(self.Nz):
            tot += np.searchsorted(self.sorted_layers[k], 0.5 * s * b[k], side="right")
        return tot / self.nvox

    def solve_scale(self, q, rho_target, tol=1e-7):
        b = bezier_profile(self.u, q)
        lo, hi = 0.0, 1.0
        while self.fraction(hi, b) < rho_target:
            hi *= 2.0
            if hi > 1e4:
                raise RuntimeError("density root not bracketed")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.fraction(mid, b) < rho_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * hi:
                break
        return hi, b

    def solid_mask(self, q, rho_target):
        s, b = self.solve_scale(q, rho_target)
        thr = 0.5 * s * b  # per layer half thickness
        mask = self.absd <= thr[None, None, :]
        return mask, s


def largest_component_periodic(mask):
    """Keep the largest face-connected component, periodic in x and y."""
    lab, n = ndimage.label(mask)  # 6-connectivity default
    if n <= 1:
        return mask
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for A, B in ((lab[0, :, :], lab[-1, :, :]), (lab[:, 0, :], lab[:, -1, :])):
        sel = (A > 0) & (B > 0)
        for i, j in zip(A[sel].ravel(), B[sel].ravel()):
            union(int(i), int(j))
    roots = np.array([find(i) for i in range(n + 1)])
    counts = np.bincount(roots[lab.ravel()], minlength=n + 1)
    counts[0] = 0
    keep = np.argmax(counts)
    return roots[lab] == keep


# --------------------------------------------------------------------------
# Mesh construction
# --------------------------------------------------------------------------

# local node offsets (standard hexahedron ordering)
OFF = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)


def build_mesh(mask, h):
    """Convert a solid voxel mask (N,N,Nz) into a periodic hexahedral mesh.

    Returns dict with nodes (nn,3), elems (ne,8), shift flags sx, sy (ne,8),
    element grid indices eijk (ne,3), surface flags, lateral periods.
    """
    N, N2, Nz = mask.shape
    assert N == N2
    ei, ej, ek = np.nonzero(mask)
    ne = ei.size
    # node grid indices for each local node
    gi = (ei[:, None] + OFF[None, :, 0])
    gj = (ej[:, None] + OFF[None, :, 1])
    gk = (ek[:, None] + OFF[None, :, 2])
    sx = (gi == N).astype(np.int8)
    sy = (gj == N).astype(np.int8)
    gi = gi % N
    gj = gj % N
    gid = (gk * N + gj) * N + gi
    used, inv = np.unique(gid.ravel(), return_inverse=True)
    elems = inv.reshape(ne, 8).astype(np.int64)
    ui = used % N
    uj = (used // N) % N
    uk = used // (N * N)
    nodes = np.stack([ui * h, uj * h, uk * h], axis=1).astype(np.float64)
    # surface elements: any of 6 face neighbours empty (z-ends not counted)
    m = mask
    exposed = np.zeros_like(m)
    exposed |= m & ~np.roll(m, 1, axis=0)
    exposed |= m & ~np.roll(m, -1, axis=0)
    exposed |= m & ~np.roll(m, 1, axis=1)
    exposed |= m & ~np.roll(m, -1, axis=1)
    up = np.zeros_like(m)
    up[:, :, :-1] = m[:, :, 1:]
    up[:, :, -1] = True
    dn = np.zeros_like(m)
    dn[:, :, 1:] = m[:, :, :-1]
    dn[:, :, 0] = True
    exposed |= m & ~up
    exposed |= m & ~dn
    surf = exposed[ei, ej, ek]
    return dict(nodes=nodes, elems=elems, sx=sx, sy=sy,
                eijk=np.stack([ei, ej, ek], axis=1).astype(np.int64),
                surf=surf.astype(np.bool_), N=N, Nz=Nz, h=h,
                Lx=N * h, Ly=N * h, H=Nz * h)


def make_design_mesh(q, rho_target=0.25, a=25.0, N=36, nz=4, grid=None):
    grid = grid or VoxelGrid(a, N, nz)
    mask, s = grid.solid_mask(q, rho_target)
    mask = largest_component_periodic(mask)
    mesh = build_mesh(mask, grid.h)
    mesh["s"] = s
    mesh["rho"] = mask.sum() / grid.nvox
    mesh["q"] = np.asarray(q, float)
    mesh["a"] = a
    return mesh, mask


# --------------------------------------------------------------------------
# Continuum (fine-grid) scale factor and minimum thickness for the
# design constraint; independent of the FE fidelity.
# --------------------------------------------------------------------------

class ContinuumDensity:
    """Layer-wise density model rho_loc(t) of a uniform sheet, tabulated on a
    fine grid; rho(theta;s) = int_0^1 rho_loc(s b(u)) du."""

    def __init__(self, a=25.0, Nfine=96, tmax=12.0, nt=600):
        g = VoxelGrid(a, Nfine, 1)
        d = np.sort(g.absd.ravel())
        self.t = np.linspace(0.0, tmax, nt)
        self.rho_loc = np.searchsorted(d, 0.5 * self.t, side="right") / d.size
        self.uq = (np.arange(400) + 0.5) / 400

    def rho(self, s, q):
        b = bezier_profile(self.uq, q)
        return np.interp(s * b, self.t, self.rho_loc).mean()

    def scale(self, q, rho_target):
        lo, hi = 1e-6, 1.0
        while self.rho(hi, q) < rho_target:
            hi *= 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.rho(mid, q) < rho_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def tmin(self, q, rho_target):
        return self.scale(q, rho_target) * bezier_min(q)

    def scale_batch(self, Q, rho_target, iters=50):
        Q = np.atleast_2d(np.asarray(Q, float))
        B = bernstein3(self.uq) @ Q.T            # (nu, n)
        lo = np.full(Q.shape[0], 1e-6); hi = np.full(Q.shape[0], 1.0)
        for _ in range(40):
            r = np.interp(hi[None, :] * B, self.t, self.rho_loc).mean(0)
            m = r < rho_target
            if not m.any():
                break
            hi[m] *= 2
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r = np.interp(mid[None, :] * B, self.t, self.rho_loc).mean(0)
            m = r < rho_target
            lo = np.where(m, mid, lo); hi = np.where(m, hi, mid)
        return 0.5 * (lo + hi)

    def tmin_batch(self, Q, rho_target):
        Q = np.atleast_2d(np.asarray(Q, float))
        s = self.scale_batch(Q, rho_target)
        return s * np.array([bezier_min(q) for q in Q])
