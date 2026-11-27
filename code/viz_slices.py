import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, _ROOT + 'code')
from geometry import make_design_mesh
def main(prefix, q, N, rho=0.25, a=25.0, keys=('0.100','0.300','0.600'), out=None, jslab=None):
    S = np.load(prefix + '_snaps.npz')
    conn, sx, sy, eijk = S['conn'], S['sx'], S['sy'], S['eijk']
    mesh, _ = make_design_mesh(q, rho, a, N, 4)
    def cents(x, L):
        xe = x[conn] + np.stack([sx * L[0], sy * L[1], np.zeros_like(sx)], -1)
        return xe.mean(1)
    j0 = N // 2 if jslab is None else jslab
    sel = (eijk[:, 1] >= j0 - 1) & (eijk[:, 1] <= j0)
    fig, axs = plt.subplots(1, len(keys) + 1, figsize=(4 * (len(keys) + 1), 5.5))
    c0 = cents(mesh['nodes'], [a, a])
    axs[0].scatter(c0[sel, 0], c0[sel, 2], s=3, c='k'); axs[0].set_title('undeformed')
    for ax, k in zip(axs[1:], keys):
        c = cents(S['x_' + k], S['L_' + k]); ep = S['ep_' + k]
        ax.scatter(c[sel, 0], c[sel, 2], s=3, c=ep[sel], vmin=0, vmax=1.0, cmap='viridis')
        ax.set_title('eps=%s Lx=%.2f' % (k, S['L_' + k][0]))
    for ax in axs:
        ax.set_aspect('equal'); ax.set_ylim(-2, 4 * a + 2); ax.set_xlim(-3, 1.3 * a)
    plt.savefig(out or prefix + '_slices.png', dpi=60, bbox_inches='tight')
if __name__ == '__main__':
    main(sys.argv[1], [float(v) for v in sys.argv[2:6]], int(sys.argv[6]))
