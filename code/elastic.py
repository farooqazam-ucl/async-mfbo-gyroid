"""Linear-elastic static homogenisation of the voxel column (same element
formulation as the explicit solver: one-point hexahedron + cube-exact
stiffness hourglass control).  Uniaxial macroscopic stress: frictionless
platens, periodic lateral faces with free lateral periods."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, _ROOT + 'code')
from geometry import make_design_mesh

XI = np.array([-1, 1, 1, -1, -1, 1, 1, -1], float)
ETA = np.array([-1, -1, 1, 1, -1, -1, 1, 1], float)
ZETA = np.array([-1, -1, -1, -1, 1, 1, 1, 1], float)
HGV = np.array([[1, -1, 1, -1, 1, -1, 1, -1], [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1], [-1, 1, -1, 1, 1, -1, 1, -1]], float)


def cube_stiffness(h, E, nu, hg_coef=1.0 / 48.0):
    dN = np.stack([XI, ETA, ZETA], 1) / (4.0 * h)  # dN/dx at centre of cube
    B = np.zeros((6, 24))
    for I in range(8):
        bx, by, bz = dN[I]
        B[0, 3 * I] = bx; B[1, 3 * I + 1] = by; B[2, 3 * I + 2] = bz
        B[3, 3 * I] = by; B[3, 3 * I + 1] = bx
        B[4, 3 * I + 1] = bz; B[4, 3 * I + 2] = by
        B[5, 3 * I] = bz; B[5, 3 * I + 2] = bx
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6)); D[:3, :3] = lam; D[np.arange(3), np.arange(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    K = h ** 3 * B.T @ D @ B
    k = hg_coef * E / (1 - nu * nu) * h
    Kh8 = k * sum(np.outer(HGV[a], HGV[a]) for a in range(4))
    K += np.kron(Kh8, np.eye(3))
    return K


def homogenise(q=(1, 1, 1, 1), rho=0.25, a=25.0, N=24, nz=4, E=1650.0, nu=0.40,
               hg_coef=1.0 / 48.0, mesh=None, tol=1e-8):
    if mesh is None:
        mesh, _ = make_design_mesh(q, rho, a, N, nz)
    h = mesh['h']; conn = mesh['elems']; sx = mesh['sx']; sy = mesh['sy']
    nn = mesh['nodes'].shape[0]; ne = conn.shape[0]
    Ke = cube_stiffness(h, E, nu, hg_coef)
    ndof = 3 * nn + 2
    # transformation: element local dof (e, 3I+c) -> global
    rows = []; cols = []; vals = []
    base = np.arange(ne)[:, None] * 24
    for I in range(8):
        for c in range(3):
            ld = base[:, 0] + 3 * I + c
            rows.append(ld); cols.append(3 * conn[:, I] + c); vals.append(np.ones(ne))
            if c == 0:
                m = sx[:, I] == 1
                rows.append(ld[m]); cols.append(np.full(m.sum(), 3 * nn)); vals.append(np.ones(m.sum()))
            if c == 1:
                m = sy[:, I] == 1
                rows.append(ld[m]); cols.append(np.full(m.sum(), 3 * nn + 1)); vals.append(np.ones(m.sum()))
    T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(24 * ne, ndof))
    Kb = sp.kron(sp.identity(ne, format='csr'), sp.csr_matrix(Ke), format='csr')
    K = (T.T @ Kb @ T).tocsr()
    z = mesh['nodes'][:, 2]; H = mesh['H']
    top = np.nonzero(np.isclose(z, H))[0]; bot = np.nonzero(np.isclose(z, 0.0))[0]
    delta = 1e-3 * H
    fixed = np.concatenate([3 * top + 2, 3 * bot + 2, [3 * bot[0], 3 * bot[0] + 1]])
    uval = np.concatenate([np.full(top.size, -delta), np.zeros(bot.size), [0.0, 0.0]])
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof); u[fixed] = uval
    rhs = -K[free][:, fixed] @ uval
    Kff = K[free][:, free]
    t0 = time.time()
    import pyamg
    # near-nullspace: rigid body modes restricted to free dofs
    X = mesh['nodes']
    B = np.zeros((ndof, 6))
    for c in range(3):
        B[c:3 * nn:3, c] = 1.0
    B[0:3 * nn:3, 3] = -X[:, 1]; B[1:3 * nn:3, 3] = X[:, 0]
    B[1:3 * nn:3, 4] = -X[:, 2]; B[2:3 * nn:3, 4] = X[:, 1]
    B[2:3 * nn:3, 5] = -X[:, 0]; B[0:3 * nn:3, 5] = X[:, 2]
    ml = pyamg.smoothed_aggregation_solver(Kff, B=B[free], max_coarse=500)
    res = []
    u[free] = ml.solve(rhs, tol=tol, accel='cg', maxiter=300, residuals=res)
    niter = len(res)
    R = K @ u
    F = -R[3 * top + 2].sum()
    A0 = mesh['Lx'] * mesh['Ly']
    Estar = F / A0 / (delta / H)
    nustar = 0.5 * (u[3 * nn] / mesh['Lx'] + u[3 * nn + 1] / mesh['Ly']) / (delta / H)
    return dict(E=Estar, nu=nustar, rho=mesh['rho'], ne=ne, solve_time=time.time() - t0, iters=niter)


if __name__ == '__main__':
    for N in [int(v) for v in sys.argv[1:]]:
        r = homogenise(N=N)
        print(N, r)
