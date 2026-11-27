"""Figures 1-3: design space, voxel pipeline, through-thickness resolution."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json
import numpy as np
sys.path.insert(0, _ROOT + 'code')
from figstyle import setup, panel, plt, C, FIGDIR, FW
from geometry import (d_app, gyroid, bezier_profile, ContinuumDensity, VoxelGrid,
                      make_design_mesh, bezier_min)
from matplotlib.colors import ListedColormap
setup()
A = 25.0; RHO = 0.25; TMIN = 1.2
cd = ContinuumDensity()


def fig1():
    fig = plt.figure(figsize=(FW, 4.6))
    # (a) gyroid surface G=0 over one unit cell (marching cubes)
    from skimage import measure
    n = 64; g = np.linspace(0, A, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    k = 2 * np.pi / A
    G = gyroid(k * X, k * Y, k * Z)
    verts, faces, _, _ = measure.marching_cubes(G, 0.0, spacing=(g[1] - g[0],) * 3)
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='#9ecae1', lw=0, alpha=1.0,
                    shade=True, rasterized=True)
    ax.set_box_aspect((1, 1, 1)); ax.set_xticks([0, 25]); ax.set_yticks([0, 25]); ax.set_zticks([0, 25])
    ax.set_xlabel('$x$ (mm)', labelpad=-8); ax.set_ylabel('$y$ (mm)', labelpad=-8); ax.set_zlabel('$z$ (mm)', labelpad=-8)
    ax.tick_params(pad=-3)
    ax.set_title(r'\textbf{(a) Gyroid surface $G=0$}', loc='left', fontsize=9, y=1.0)
    # (b) sheet band on a section y=a/4 from normalised distance
    ax = fig.add_subplot(2, 2, 2)
    m = 400; xs = np.linspace(0, A, m); zs = np.linspace(0, A, m)
    XX, ZZ = np.meshgrid(xs, zs, indexing='xy')
    d = d_app(XX, np.full_like(XX, A / 4), ZZ, A)
    t = 2.09
    ax.contourf(XX, ZZ, (np.abs(d) <= t / 2).astype(float), levels=[0.5, 1.5], colors=['#bdbdbd'])
    ax.contour(XX, ZZ, d, levels=[0.0], colors='k', linestyles='--', linewidths=0.8)
    ax.set_aspect('equal'); ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$z$ (mm)')
    panel(ax, '(b) Sheet section at $y=a/4$')
    # (c) Bezier profiles from the design of experiments
    ax = fig.add_subplot(2, 2, 3)
    D = json.load(open(_ROOT + 'data/doe_designs.json'))
    u = np.linspace(0, 1, 200)
    for i, q in enumerate(D['lf'][:36]):
        s = cd.scale(q, RHO)
        ax.plot(s * bezier_profile(u, q), u * 100, color=C['grey'], lw=0.5, alpha=0.6)
    s = cd.scale([1, 1, 1, 1], RHO)
    ax.plot(s * bezier_profile(u, [1, 1, 1, 1]), u * 100, color='k', lw=1.5, label='uniform')
    for q, col in (([1.35, 1.1, 0.85, 0.7], C['B']), ([0.75, 1.3, 1.25, 0.7], C['A'])):
        qq = np.array(q) / np.mean(q)
        s = cd.scale(qq, RHO)
        ax.plot(s * bezier_profile(u, qq), u * 100, color=col, lw=1.3)
    ax.axvline(TMIN, color='r', ls=':', lw=1.0); ax.text(TMIN - 0.06, 50, '$t_{\\min}$', color='r', ha='right', va='center')
    ax.set_xlabel('wall thickness $t$ (mm)'); ax.set_ylabel('height $z$ (mm)'); ax.set_xlim(0.8, 3.8)
    ax.legend(loc='lower right', frameon=False)
    panel(ax, '(c) Thickness profiles')
    # (d) mid-plane section of a graded column
    ax = fig.add_subplot(2, 2, 4)
    q = np.array([1.35, 1.1, 0.85, 0.7]); q = q / q.mean()
    s = cd.scale(q, RHO)
    xs = np.linspace(0, A, 300); zs = np.linspace(0, 4 * A, 1200)
    XX, ZZ = np.meshgrid(xs, zs, indexing='xy')
    d = d_app(XX, np.full_like(XX, A / 2), ZZ, A)
    tt = s * bezier_profile(ZZ / (4 * A), q)
    ax.contourf(XX, ZZ, (np.abs(d) <= tt / 2).astype(float), levels=[0.5, 1.5], colors=['#4a4a4a'])
    ax.set_aspect('equal'); ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$z$ (mm)')
    ax.set_xlim(0, A); ax.set_ylim(0, 4 * A)
    panel(ax, '(d) Graded column')
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.42, hspace=0.5)
    fig.savefig(FIGDIR + 'fig01_geometry_parameterisation.pdf')


def fig2():
    fig, axs = plt.subplots(1, 4, figsize=(FW, 2.1))
    N = 24; h = A / N
    xs = np.linspace(0, A, 400); zs = np.linspace(0, A, 400)
    XX, ZZ = np.meshgrid(xs, zs, indexing='xy')
    d = d_app(XX, np.full_like(XX, (N // 2 + 0.5) * h), ZZ, A)
    axs[0].imshow(d, origin='lower', extent=(0, A, 0, A), cmap='RdBu_r', vmin=-4, vmax=4)
    axs[0].contour(XX, ZZ, d, levels=[0], colors='k', linewidths=0.6)
    panel(axs[0], '(a) $d_{\\mathrm{app}}$')
    c = (np.arange(N) + 0.5) * h
    CX, CZ = np.meshgrid(c, c, indexing='xy')
    dc = d_app(CX, np.full_like(CX, (N // 2 + 0.5) * h), CZ, A)
    t = 2.09
    solid = np.abs(dc) <= t / 2
    axs[1].contour(XX, ZZ, d, levels=[-t / 2, t / 2], colors='k', linewidths=0.6)
    axs[1].scatter(CX[~solid], CZ[~solid], s=1.2, c='#bbbbbb'); axs[1].scatter(CX[solid], CZ[solid], s=2.5, c='k')
    panel(axs[1], '(b) Centroids')
    axs[2].imshow(solid, origin='lower', extent=(0, A, 0, A), cmap=ListedColormap(['white', '#555555']))
    axs[2].contour(XX, ZZ, d, levels=[-t / 2, t / 2], colors='#c0392b', linewidths=0.7)
    panel(axs[2], '(c) Voxels')
    # zoom
    axs[3].imshow(solid, origin='lower', extent=(0, A, 0, A), cmap=ListedColormap(['white', '#555555']))
    axs[3].contour(XX, ZZ, d, levels=[-t / 2, t / 2], colors='#c0392b', linewidths=0.9)
    for v in np.arange(0, A + 1e-9, h):
        axs[3].axvline(v, color='#9ecae1', lw=0.3); axs[3].axhline(v, color='#9ecae1', lw=0.3)
    axs[3].set_xlim(4, 12); axs[3].set_ylim(8, 16)
    panel(axs[3], '(d) Detail')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    fig.savefig(FIGDIR + 'fig02_voxel_pipeline.pdf')


def fig3():
    Ns = [16, 24, 36, 48]
    fig, axs = plt.subplots(1, 4, figsize=(FW, 1.95))
    t = 2.09
    for ax, N in zip(axs, Ns):
        h = A / N
        c = (np.arange(N) + 0.5) * h
        CX, CZ = np.meshgrid(c, c, indexing='xy')
        dc = d_app(CX, np.full_like(CX, (N // 2 + 0.5) * h), CZ, A)
        solid = np.abs(dc) <= t / 2
        ax.imshow(solid, origin='lower', extent=(0, A, 0, A), cmap=ListedColormap(['white', '#606060']))
        xs = np.linspace(0, A, 500); XX, ZZ = np.meshgrid(xs, xs, indexing='xy')
        d = d_app(XX, np.full_like(XX, (N // 2 + 0.5) * h), ZZ, A)
        ax.contour(XX, ZZ, d, levels=[-t / 2, t / 2], colors='k', linewidths=0.6)
        ax.set_xlim(2, 14); ax.set_ylim(6, 18)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        lab = {24: ' (LF)', 36: ' (HF)'}.get(N, '')
        ax.set_title('$N=%d$%s\n$\\bar t/h=%.1f$' % (N, lab, t / h), fontsize=8.5)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.08)
    fig.savefig(FIGDIR + 'fig03_through_thickness_resolution.pdf')


if __name__ == '__main__':
    for f in sys.argv[1:]:
        globals()[f]()
