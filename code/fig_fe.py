"""Figures based on FE results: 4-8, 16, S1."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, glob
import numpy as np
sys.path.insert(0, _ROOT + 'code')
from figstyle import setup, panel, plt, C, FIGDIR, FW
from metrics import curve_metrics, cumulative_energy, first_crossing
from analyse_fe import load, record
setup()
RUNS = _ROOT + 'runs/'
DATA = _ROOT + 'data/'


def mid_section(ax, prefix, key, a=25.0, N=None, vmax=0.8, cmap='viridis', jfrac=0.5):
    S = np.load(RUNS + prefix + '_snaps.npz')
    conn, sx, sy, eijk = S['conn'], S['sx'], S['sy'], S['eijk']
    x = S['x_' + key]; L = S['L_' + key]; ep = S['ep_' + key]
    Nn = eijk[:, 0].max() + 1
    xe = x[conn] + np.stack([sx * L[0], sy * L[1], np.zeros_like(sx)], -1)
    j0 = int(jfrac * Nn)
    sel = (eijk[:, 1] == j0)
    # draw each element of the slab as the quadrilateral of its y-mid face projected on x-z
    from matplotlib.collections import PolyCollection
    quads = []
    for e in np.nonzero(sel)[0]:
        P = xe[e]
        # mid-plane of the element in y: average of nodes (0,3),(1,2),(5,6),(4,7)
        q = np.array([(P[0] + P[3]) / 2, (P[1] + P[2]) / 2, (P[5] + P[6]) / 2, (P[4] + P[7]) / 2])
        quads.append(q[:, [0, 2]])
    pc = PolyCollection(quads, array=ep[sel], cmap=cmap, edgecolors='none')
    pc.set_clim(0, vmax)
    ax.add_collection(pc)
    pc.set_clip_path(ax.patch)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, L[0] + 0.5)
    ax.set_ylim(-1, 4 * a + 1)
    ax.set_xticks([]); ax.set_yticks([])
    return pc


def fig4():
    Ns = [16, 24, 30, 36, 48]
    pre = {16: 'conv/u_N16', 24: 'doe/lf_00', 30: 'conv/u_N30', 36: 'doe/hf_00', 48: 'conv/u_N48'}
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.6))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(Ns)))
    M = {}
    for N, c in zip(Ns, cols):
        d, m = load(RUNS + pre[N])
        axs[0].plot(d['strain'], d['stress'], color=c, lw=1.0, label='$N=%d$' % N)
        M[N] = record(RUNS + pre[N])
    axs[0].set_xlabel('nominal strain $\\varepsilon$'); axs[0].set_ylabel('nominal stress $\\sigma$ (MPa)')
    axs[0].legend(ncol=2, frameon=False); axs[0].set_xlim(0, 0.6); axs[0].set_ylim(0, None)
    panel(axs[0], '(a) Uniform design')
    ref = M[48]
    tdh = [2.09 * N / 25.0 for N in Ns]
    for key, lab, mk in (('SEA', 'SEA', 'o'), ('ip', '$\\sigma_{\\mathrm{ip}}$', 's'), ('plat_mean', '$\\bar\\sigma_p$', '^')):
        axs[1].plot(tdh, [M[N][key] / ref[key] for N in Ns], marker=mk, ms=4, label=lab)
    axs[1].axhspan(0.95, 1.05, color='0.9', zorder=0)
    for N, lab in ((24, 'LF'), (36, 'HF')):
        axs[1].axvline(2.09 * N / 25, color='k', ls=':', lw=0.8)
        axs[1].text(2.09 * N / 25 + 0.05, axs[1].get_ylim()[1] if False else 1.17, lab, fontsize=7)
    axs[1].set_xlabel('elements through mean thickness $\\bar t/h$'); axs[1].set_ylabel('value / value($N=48$)')
    axs[1].legend(frameon=False); axs[1].set_ylim(0.85, 1.2)
    panel(axs[1], '(b) Normalised responses')
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIGDIR + 'fig04_mesh_convergence.pdf')
    return M


def fig5():
    E = json.load(open(DATA + 'elastic.json'))
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.6))
    conv = E['conv']
    h = np.array([c['h'] for c in conv]); Es = np.array([c['E_adj'] for c in conv]); Ns = np.array([c['N'] for c in conv])
    sel = Ns >= 30
    axs[0].axhspan(Es[sel].min(), Es[sel].max(), color='0.88', zorder=0, label='range, $N\\geq30$')
    axs[0].plot(h, Es, 'o-', color=C['B'], lw=0.8, ms=4, label='voxel model')
    offs = {56: (-6, 6), 48: (0, 7), 42: (-7, -12), 36: (6, -12), 30: (4, -12), 24: (4, 5), 20: (6, -6), 16: (5, -10)}
    for hi, Ni, Ei in zip(h, Ns, Es):
        axs[0].annotate('%d' % Ni, (hi, Ei), textcoords='offset points', xytext=offs.get(int(Ni), (3, 4)),
                        ha='center')
    axs[0].axhline(892 * 0.25 ** 1.39, color='k', ls='--', lw=0.9, label='experimental fit')
    for N, lab in ((24, 'LF'), (36, 'HF')):
        axs[0].axvline(25.0 / N, color='k', ls=':', lw=0.7)
        axs[0].text(25.0 / N + 0.01, 120, lab, fontsize=7)
    axs[0].set_xlabel('voxel size $h$ (mm)'); axs[0].set_ylabel('effective modulus $E^\\ast$ (MPa)')
    axs[0].set_xlim(0, 1.7); axs[0].set_ylim(115, 165); axs[0].legend(frameon=False, fontsize=6.5, loc='upper left')
    panel(axs[0], '(a) Uniform design (labels: $N$)')
    dens = E['density']
    r = np.array([d['rho'] for d in dens]); e = np.array([d['E'] for d in dens])
    rr = np.linspace(0.07, 0.46, 100)
    axs[1].loglog(rr, 892 * rr ** 1.39, 'k--', lw=0.9, label='experimental fit')
    axs[1].loglog(r, e, 'o', color=C['B'], ms=4, label='voxel model, $N=36$')
    if E.get('density_fine'):
        df = E['density_fine']
        axs[1].loglog([d['rho'] for d in df], [d['E'] for d in df], 's', mfc='none', color=C['A'], ms=5,
                      label='voxel model, $N=48$--$64$')
    axs[1].set_xlabel('relative density $\\bar\\rho$'); axs[1].set_ylabel('$E^\\ast$ (MPa)')
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    axs[1].xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.3, 0.4])); axs[1].xaxis.set_minor_locator(NullLocator())
    axs[1].xaxis.set_major_formatter(FixedFormatter(['0.1', '0.2', '0.3', '0.4']))
    axs[1].yaxis.set_major_locator(FixedLocator([30, 50, 100, 200, 300])); axs[1].yaxis.set_minor_locator(NullLocator())
    axs[1].yaxis.set_major_formatter(FixedFormatter(['30', '50', '100', '200', '300']))
    axs[1].legend(frameon=False, fontsize=6.5)
    panel(axs[1], '(b) Modulus--density scaling')
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(FIGDIR + 'fig05_voxel_discretisation.pdf')


def fig6():
    E = json.load(open(DATA + 'elastic.json'))
    fig = plt.figure(figsize=(FW, 5.0))
    gtop = fig.add_gridspec(1, 2, left=0.11, right=0.985, top=0.95, bottom=0.62, wspace=0.42)
    ax = fig.add_subplot(gtop[0])
    runs = [('val/r305_N36', 0.305, C['A']), ('doe/hf_00', 0.25, C['B']), ('val/r140_N48', 0.14, C['C'])]
    V = []
    for pre, rho, col in runs:
        d, m = load(RUNS + pre)
        ax.plot(d['strain'], d['stress'], color=col, label='$\\bar\\rho=%g$' % rho)
        S = 54 * m['rho'] ** 1.63
        ax.plot([0.0, 0.6], [S, S], ls='--', color=col, lw=0.7)
        V.append(dict(rho=m['rho'], S=float(np.max(d['stress'][d['strain'] <= 0.3])), Sexp=S))
    ax.set_xlabel('nominal strain'); ax.set_ylabel('nominal stress (MPa)'); ax.set_xlim(0, 0.6); ax.set_ylim(0, 9)
    ax.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.42), fontsize=7, labelspacing=0.3); panel(ax, '(a) Stress--strain responses')
    ax = fig.add_subplot(gtop[1])
    rr = np.linspace(0.07, 0.46, 100)
    ax.loglog(rr, 54 * rr ** 1.63, 'k--', lw=0.9, label='$\\sigma^\\ast$ fit')
    ax.loglog(rr, 892 * rr ** 1.39 / 20, 'k:', lw=0.9, label='$E^\\ast/20$ fit')
    ax.loglog([v['rho'] for v in V], [v['S'] for v in V], 'o', color=C['A'], label='$\\sigma^\\ast$ model')
    ax.loglog([d['rho'] for d in E['density']], [d['E'] / 20 for d in E['density']], 's', ms=3.5, color=C['B'], label='$E^\\ast/20$ model')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.3, 0.4])); ax.xaxis.set_major_formatter(FixedFormatter(['0.1', '0.2', '0.3', '0.4']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('relative density'); ax.set_ylabel('stress or modulus/20 (MPa)'); ax.legend(frameon=False, ncol=1, loc='upper left', labelspacing=0.25)
    panel(ax, '(b) Strength and modulus')
    vmax = 0.4
    figh = 5.0
    items = [('val/r305_N36', '0.200', '$\\varepsilon=0.2$'), ('val/r305_N36', '0.400', '$\\varepsilon=0.4$'),
             ('val/r140_N48', '0.200', '$\\varepsilon=0.2$'), ('val/r140_N48', '0.400', '$\\varepsilon=0.4$')]
    sax, cax, xr = section_row(fig, figh, items, vmax=vmax)
    for k, rl in ((0, '0.305'), (2, '0.14')):
        sax[k].set_title(('\\textbf{(c)} ' if k == 0 else '') + '$\\bar\\rho=%s$' % rl, loc='left', fontsize=9)
    x0 = xr + 0.9
    axp = fig.add_axes([x0 / FW, 0.42 / figh, (FW - x0 - 0.05) / FW, 1.9 / figh])
    sc = plt.cm.viridis(np.linspace(0.0, 0.85, 5))
    for i, key in enumerate(('0.100', '0.200', '0.300', '0.400', '0.500')):
        z, p = layer_profile('val/r140_N48', key); axp.plot(p, z * 100, '-o', color=sc[i], ms=2.5, lw=1.0,
                                                           label='$\\varepsilon=%s$' % key.rstrip('0'))
        z, p = layer_profile('val/r305_N36', key); axp.plot(p, z * 100, '--', color=sc[i], lw=0.8)
    axp.set_xlabel('mean $\\bar\\varepsilon^p$ in slab'); axp.set_ylabel('undeformed height (mm)')
    axp.set_ylim(0, 100); axp.set_xlim(0, 0.5)
    axp.legend(frameon=False, loc='center right', handlelength=1.2, labelspacing=0.25)
    panel(axp, '(d) Slab-mean plastic strain')
    fig.savefig(FIGDIR + 'fig06_experimental_validation.pdf')
    json.dump(V, open(DATA + 'validation.json', 'w'), indent=1)
    return V


def fig7(graded='doe/hf_07'):
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.6))
    for pre, col, lab in (('doe/hf_00', C['uni'], 'uniform'), (graded, C['A'], 'graded')):
        d, m = load(RUNS + pre); e, s = d['strain'], d['stress']
        r = record(RUNS + pre)
        axs[0].plot(e, s, color=col, label=lab)
        W = cumulative_energy(e, s)
        k = np.ones(5) / 5; ss = np.convolve(s, k, 'same')
        eff = W / np.maximum(ss, 1e-9)
        axs[1].plot(e, eff, color=col, label=lab)
        axs[1].axvline(r['eps_d'], color=col, ls=':', lw=0.8)
        i_ip = np.argmax(s[e <= 0.1]); axs[0].plot(e[i_ip], s[i_ip], 'v', color=col, ms=5)
        i_mx = np.argmax(s); axs[0].plot(e[i_mx], s[i_mx], '^', color=col, ms=5)
        axs[0].hlines(r['plat_mean'], 0.1, 0.5, colors=col, linestyles='--', lw=0.8)
    axs[0].axvspan(0.1, 0.5, color='0.93', zorder=0)
    axs[0].axhline(SIGLIM, color='r', lw=0.8, ls='-.')
    axs[0].text(0.01, SIGLIM + 0.15, '$\\sigma_{\\mathrm{limit}}$', color='r', fontsize=7)
    axs[0].text(0.3, 0.4, 'plateau window', ha='center', color='0.4')
    axs[0].set_xlabel('nominal strain'); axs[0].set_ylabel('nominal stress (MPa)'); axs[0].set_xlim(0, 0.6); axs[0].set_ylim(0, None)
    axs[0].legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.33))
    panel(axs[0], '(a) $\\sigma_{\\mathrm{ip}}$ ($\\triangledown$), $\\sigma_{\\max}$ ($\\triangle$), $\\bar\\sigma_p$ (dashed)')
    axs[1].set_xlabel('nominal strain'); axs[1].set_ylabel('efficiency $\\eta_E=W_v/\\sigma$'); axs[1].set_xlim(0, 0.6)
    axs[1].legend(frameon=False); panel(axs[1], '(b) Efficiency and $\\varepsilon_d$ (dotted)')
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIGDIR + 'fig07_metric_definitions.pdf')


def fig8(pruned='prune/hf_demo', full='doe/hf_07'):
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.6))
    dF, _ = load(RUNS + full); dP, mP = load(RUNS + pruned)
    axs[0].plot(dF['strain'], dF['stress'], color='0.75', lw=1.0, label='complete response')
    axs[0].plot(dP['strain'], dP['stress'], color=C['B'], lw=1.0, label='pruned analysis')
    ma = np.convolve(dP['stress'], np.ones(5) / 5, 'valid')
    axs[0].plot(dP['strain'][4:], ma, color='k', lw=0.8, ls='--', label='5-sample mean')
    axs[0].axhline(SIGLIM, color='r', lw=0.8, ls='-.'); axs[0].text(0.01, SIGLIM + 0.2, '$\\sigma_{\\mathrm{limit}}$', color='r', fontsize=7)
    axs[0].axvline(dP['strain'][-1], color=C['B'], ls=':', lw=0.8)
    axs[0].set_xlabel('nominal strain'); axs[0].set_ylabel('nominal stress (MPa)'); axs[0].set_xlim(0, 0.6); axs[0].set_ylim(0, None)
    axs[0].legend(frameon=False, loc='lower right'); panel(axs[0], '(a) Pruning at $\\varepsilon=%.3f$' % dP['strain'][-1])
    clean = json.load(open(DATA + 'clean_timing.json'))
    for f in sorted(glob.glob(RUNS + 'doe/hf_*_hist.npz')):
        d = np.load(f)
        if 'wall_hist' in d and clean.get(f.split('/')[-1][:5], False):
            axs[1].plot(d['strain'], d['wall_hist'] / 60.0, color='0.7', lw=0.6)
    axs[1].plot(dF['strain'], dF['wall_hist'] / 60.0, color='k', lw=1.0)
    ec = dP['strain'][-1]
    wc = np.interp(ec, dF['strain'], dF['wall_hist']) / 60.0
    axs[1].annotate('', xy=(0.6, dF['wall_hist'][-1] / 60.0), xytext=(0.6, wc),
                    arrowprops=dict(arrowstyle='<->', color=C['B']))
    axs[1].plot([ec, 0.6], [wc, wc], ':', color=C['B'])
    axs[1].text(0.45, (wc + dF['wall_hist'][-1] / 60.0) / 2, 'saved', color=C['B'], fontsize=7)
    axs[1].set_xlabel('nominal strain'); axs[1].set_ylabel('cumulative solve time (min)'); axs[1].set_xlim(0, 0.62)
    panel(axs[1], '(b) Cumulative solve time')
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIGDIR + 'fig08_pruning_in_action.pdf')


def figS1():
    fig, axs = plt.subplots(1, 3, figsize=(FW, 2.2))
    for f in sorted(glob.glob(RUNS + 'doe/hf_*_hist.npz')) + sorted(glob.glob(RUNS + 'test/hf_*_hist.npz')):
        d = np.load(f); e = d['strain']; tt = e / e[-1]
        axs[0].plot(tt, d['KE'] / np.maximum(d['IE'], 1e-12), color='0.5', lw=0.5)
        axs[1].plot(e, d['HG'] / np.maximum(d['IE'], 1e-12), color='0.5', lw=0.5)
        axs[2].plot(e, 100 * d['added_mass'], color='0.5', lw=0.5)
    axs[0].axhline(0.05, color='r', ls='--', lw=0.8); axs[0].axvline(0.2, color='k', ls=':', lw=0.8)
    axs[0].set_ylim(0, 0.2); axs[0].set_xlabel('$t/T$'); axs[0].set_ylabel('$E_K/E_I$'); panel(axs[0], '(a) Kinetic energy')
    axs[1].set_xlabel('nominal strain'); axs[1].set_ylabel('$E_{\\mathrm{hg}}/E_I$'); axs[1].set_ylim(0, 0.2); panel(axs[1], '(b) Hourglass energy')
    axs[2].set_xlabel('nominal strain'); axs[2].set_ylabel('added mass (\\%)'); panel(axs[2], '(c) Added mass')
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(FIGDIR + 'figS1_energy_diagnostics.pdf')




def section_row(fig, figh, items, x0=0.1, y0=0.42, h=1.9, gap=0.13, vmax=0.8, cbar_gap=0.16):
    """Draw a row of mid-plane sections (items: list of (prefix, key, label)) with
    explicit positions in inches; returns (axes list, colourbar axes, right edge in inches)."""
    w = h / 4.0 * (25.0 + 1.0) / 102.0 * 4.0  # aspect of the 25 mm x 100 mm column
    axs = []
    x = x0
    for pre, key, lab in items:
        ax = fig.add_axes([x / FW, y0 / figh, w / FW, h / figh])
        pc = mid_section(ax, pre, key, vmax=vmax)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(lab, labelpad=2)
        axs.append(ax)
        x += w + gap
    cax = fig.add_axes([(x + cbar_gap - gap) / FW, y0 / figh, 0.09 / FW, h / figh])
    fig.colorbar(pc, cax=cax, extend=('max' if vmax < 0.8 else 'neither'))
    cax.set_title('$\\bar\\varepsilon^p$', loc='left', fontsize=9)
    return axs, cax, x + cbar_gap - gap + 0.09


def layer_profile(prefix, key, nb=10):
    """Mean equivalent plastic strain in nb slabs of the undeformed height."""
    S = np.load(RUNS + prefix + '_snaps.npz')
    eijk = S['eijk']; nz = eijk[:, 2].max() + 1
    zb = (eijk[:, 2] + 0.5) / nz
    idx = np.minimum((zb * nb).astype(int), nb - 1)
    ep = S['ep_' + key]
    return (np.arange(nb) + 0.5) / nb, np.array([ep[idx == i].mean() for i in range(nb)])


def fig16():
    CR = json.load(open(DATA + 'confirm_results.json'))
    from geometry import bezier_profile, ContinuumDensity
    cd = ContinuumDensity()
    fig = plt.figure(figsize=(FW, 5.0))
    gtop = fig.add_gridspec(1, 2, left=0.11, right=0.985, top=0.95, bottom=0.62, wspace=0.38)
    ax0 = fig.add_subplot(gtop[0]); ax1 = fig.add_subplot(gtop[1])
    cols = {'uniform': 'k', 'maxSEA': C['A'], 'knee': C['C'], 'minIP': C['B'], 'minVAR': C['grey']}
    labs = {'uniform': 'uniform', 'maxSEA': 'max. SEA', 'knee': 'knee', 'minIP': 'min. $\\sigma_{\\mathrm{ip}}$', 'minVAR': 'min. $\\sigma_{\\mathrm{plat}}$'}
    u = np.linspace(0, 1, 200)
    for k in ('uniform', 'maxSEA', 'knee', 'minIP', 'minVAR'):
        if k not in CR: continue
        q = np.array(CR[k]['q']); s = cd.scale(q, 0.25)
        ax0.plot(u * 100, s * bezier_profile(u, q), color=cols[k], label=labs[k])
        d, m = load(RUNS + CR[k]['run'])
        ax1.plot(d['strain'], d['stress'], color=cols[k], lw=1.0, label=labs[k])
    ax0.axhline(1.2, color='r', ls=':', lw=0.8)
    ax0.text(99, 1.23, '$t_{\\min}$', color='r', fontsize=7, ha='right', va='bottom')
    ax0.set_ylim(1.0, 3.75)
    ax0.set_xlabel('height $z$ (mm)'); ax0.set_ylabel('wall thickness (mm)')
    ax0.legend(frameon=False, fontsize=6.5, ncol=3, loc='upper left', handlelength=1.5, columnspacing=1.0)
    panel(ax0, '(a) Thickness profiles')
    ax1.axhline(SIGLIM, color='r', ls='-.', lw=0.8)
    ax1.set_xlabel('nominal strain'); ax1.set_ylabel('nominal stress (MPa)'); ax1.set_xlim(0, 0.6); ax1.set_ylim(0, 8)
    panel(ax1, '(b) High-fidelity responses')
    run = CR['minIP']['run']
    figh = 5.0
    items = [(run, k, '$\\varepsilon=%s$' % k.rstrip('0')) for k in ('0.100', '0.200', '0.300', '0.400')]
    sax, cax, xr = section_row(fig, figh, items, vmax=0.8)
    sax[0].set_title('\\textbf{(c)}', loc='left', fontsize=9)
    x0 = xr + 0.9
    axp = fig.add_axes([x0 / FW, 0.42 / figh, (FW - x0 - 0.05) / FW, 1.9 / figh])
    sc = plt.cm.viridis(np.linspace(0.0, 0.85, 5))
    for i, key in enumerate(('0.100', '0.200', '0.300', '0.400', '0.500')):
        z, p = layer_profile(run, key); axp.plot(p, z * 100, '-o', color=sc[i], ms=2.5, lw=1.0,
                                                 label='$\\varepsilon=%s$' % key.rstrip('0'))
        z, p = layer_profile('doe/hf_00', key); axp.plot(p, z * 100, '--', color=sc[i], lw=0.8)
    axp.set_xlabel('mean $\\bar\\varepsilon^p$ in slab'); axp.set_ylabel('undeformed height (mm)')
    axp.set_ylim(0, 100); axp.set_xlim(0, 0.62)
    axp.legend(frameon=False, loc='center right', handlelength=1.2, labelspacing=0.25)
    panel(axp, '(d) Slab-mean plastic strain')
    fig.savefig(FIGDIR + 'fig16_collapse_mechanism.pdf')


SIGLIM = None
if __name__ == '__main__':
    SIGLIM = float(json.load(open(DATA + 'settings.json'))['sigma_limit'])
    for f in sys.argv[1:]:
        globals()[f]()
