"""Campaign figures: 9, 11-15, S2, S3."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, os, json, glob
import numpy as np
sys.path.insert(0, _ROOT + 'code')
from figstyle import setup, panel, plt, C, FIGDIR, FW
setup()
DATA = _ROOT + 'data/'
ARM_LABEL = {'A': 'A (sync)', 'B': 'B (async)', 'C': 'C (elastic)', 'A0': 'A, no pruning',
             'B0': 'B, no pruning', 'Bc': 'B, cold start'}
ARM_COL = {'A': C['A'], 'B': C['B'], 'C': C['C'], 'A0': C['A0'], 'B0': C['B0'], 'Bc': C['Bc'],
           'A0c': C['A0c'], 'S': C['S'], 'Sb': C['Sb'], 'R': C['R']}


def load_runs():
    R = {}
    for f in sorted(glob.glob(DATA + 'campaigns/*.json')):
        r = json.load(open(f)); R.setdefault(r['arm'], []).append(r)
    return R


def hv_on_grid(trace, grid, key):
    t = np.array([x[0] if key == 'time' else x[1] for x in trace]); hv = np.array([x[2] for x in trace])
    o = np.argsort(t, kind='stable'); t = t[o]; hv = np.maximum.accumulate(hv[o])
    return np.array([hv[t <= g].max() if np.any(t <= g) else 0.0 for g in grid])


def fig12(R):
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.6))
    ng = np.arange(0, 101)
    tmax = max(max(r['T_wall'] for r in R[a]) for a in ('A', 'B', 'C') if a in R) / 3600.0
    tg = np.linspace(0, tmax, 200)
    for a in ('A', 'B', 'C'):
        if a not in R: continue
        H1 = np.array([hv_on_grid(r['trace'], ng, 'n') for r in R[a]])
        H2 = np.array([hv_on_grid(r['trace'], tg * 3600, 'time') for r in R[a]])
        for ax, G, H in ((axs[0], ng, H1), (axs[1], tg, H2)):
            ax.plot(G, np.median(H, 0), color=ARM_COL[a], label=ARM_LABEL[a])
            ax.fill_between(G, np.percentile(H, 25, 0), np.percentile(H, 75, 0), color=ARM_COL[a], alpha=0.2, lw=0)
    axs[0].set_xlabel('evaluations returned'); axs[0].set_ylabel('hypervolume'); axs[0].legend(frameon=False)
    panel(axs[0], '(a) Sample efficiency')
    axs[1].set_xlabel('wall-clock time (h)'); axs[1].set_ylabel('hypervolume')
    panel(axs[1], '(b) Rate of discovery')
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIGDIR + 'fig12_hypervolume_convergence.pdf')


def fig13(R, seed=0):
    """Worker occupancy (Gantt chart) of one campaign per execution mode."""
    S = json.load(open(DATA + 'campaign_stats.json'))['stats']
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(4, 1, figsize=(FW, 6.0), sharex=True,
                            gridspec_kw=dict(height_ratios=[4, 4, 8, 3], hspace=0.6))
    for k, (ax, a) in enumerate(zip(axs[:3], ('A', 'B', 'C'))):
        r = [x for x in R[a] if x['seed'] == seed][0]
        nW = 8 if a == 'C' else 4
        for (wk, t0, t1, lam, st) in r['occupancy']:
            col = C['hf'] if lam == 1 else C['lf']
            hatch = '////' if st == 'pruned' else None
            ax.barh(wk, (t1 - t0) / 3600, left=t0 / 3600, height=0.72, color=col,
                    edgecolor='white' if not hatch else 'k', hatch=hatch, lw=0.3)
        ax.set_yticks(range(nW)); ax.set_yticklabels(['W%d' % (i + 1) for i in range(nW)])
        ax.set_ylim(-0.6, nW - 0.4)
        util = S[a]['utilisation']['vals'][seed]
        panel(ax, '(%s) Mode %s' % ('abc'[k], a))
        ax.text(1.0, 1.03, '$T_{\\mathrm{wall}}=%.2f$ h, utilisation %.0f\\%%' % (r['T_wall'] / 3600, 100 * util),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
        if a == 'C':
            tr = np.array(r['W_trace'] + [[r['T_wall'], r['W_trace'][-1][1]]], dtype=float)
            axs[3].step(tr[:, 0] / 3600, tr[:, 1], where='post', color=C['C'], lw=1.0)
            axs[3].set_ylim(0, 8.8); axs[3].set_yticks([2, 4, 6, 8]); axs[3].set_ylabel('$W(t)$')
            panel(axs[3], '(d) Pool size chosen by the elastic rule in Mode C')
    axs[-1].set_xlabel('wall-clock time (h)')
    handles = [Patch(color=C['lf'], label='low fidelity'), Patch(color=C['hf'], label='high fidelity'),
               Patch(facecolor='white', edgecolor='k', hatch='////', label='pruned')]
    fig.legend(handles=handles, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.955),
               columnspacing=1.5, handlelength=1.5)
    fig.savefig(FIGDIR + 'fig13_worker_occupancy.pdf')


def fig14(R):
    fig, axs = plt.subplots(1, 3, figsize=(FW, 2.4))
    arms = [a for a in ('A0', 'A', 'B0', 'B') if a in R]
    for k, (key, lab) in enumerate((('hv_final', 'terminal hypervolume'), ('cpu_hours', 'CPU time (h)'),
                                     ('T_wall', 'wall-clock time (h)'))):
        ax = axs[k]
        for i, a in enumerate(arms):
            v = np.array([r[key] for r in R[a]]) / (3600.0 if key == 'T_wall' else 1.0)
            ax.boxplot([v], positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color='k'), boxprops=dict(color=ARM_COL[a]), whiskerprops=dict(color=ARM_COL[a]),
                       capprops=dict(color=ARM_COL[a]))
            ax.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, 'o', ms=3, color=ARM_COL[a])
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([a for a in arms]); ax.set_ylabel(lab)
        panel(ax, '(%s)' % 'abc'[k])
    fig.subplots_adjust(wspace=0.45)
    fig.savefig(FIGDIR + 'fig14_h2_ablation.pdf')


def fig15(R, settings, confirm=None):
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.75))
    Y = []
    for a in [x for x in R if x in ('A', 'B', 'C', 'A0', 'B0', 'Bc')]:
        for r in R[a]:
            for o in r['obs']:
                if o['lam'] == 1 and o['status'] == 'completed' and o['feasible']:
                    Y.append([o['y'][0], -o['y'][1], -o['y'][2]])
    Y = np.array(Y)
    import torch
    from botorch.utils.multi_objective.pareto import is_non_dominated
    nd = is_non_dominated(torch.as_tensor(np.column_stack([Y[:, 0], -Y[:, 1], -Y[:, 2]]))).numpy()
    sc = axs[0].scatter(Y[:, 1], Y[:, 0], c=Y[:, 2], s=6, cmap='viridis', alpha=0.5, lw=0)
    axs[0].scatter(Y[nd, 1], Y[nd, 0], c=Y[nd, 2], s=22, cmap='viridis', edgecolors='k', lw=0.6,
                   vmin=Y[:, 2].min(), vmax=Y[:, 2].max())
    recs = json.load(open(DATA + 'fe_records.json'))
    u = [r for r in recs if r['name'] == 'doe/hf_00'][0]
    axs[0].plot(u['ip'], u['SEA'], '*', color='r', ms=11, mec='k', label='uniform (FE)')
    if confirm is None and os.path.exists(DATA + 'confirm_results.json'):
        confirm = [c for k, c in json.load(open(DATA + 'confirm_results.json')).items()
                   if k != 'uniform' and c.get('emulator')]
    if confirm:
        for i, c in enumerate(confirm):
            em = c['emulator']
            for ax, xf, xe in ((axs[0], c['ip'], -em[1]), (axs[1], np.log10(c['plat_var']), -em[2])):
                ax.plot([xe, xf], [em[0], c['SEA']], '-', color='0.35', lw=0.6, zorder=4)
                ax.plot(xe, em[0], 'D', mfc='none', mec='0.35', ms=4.5, zorder=5,
                        label=('emulator prediction' if (i == 0 and ax is axs[0]) else None))
                ax.plot(xf, c['SEA'], 'D', color='k', ms=4.5, zorder=6,
                        label=('re-analysed (FE)' if (i == 0 and ax is axs[0]) else None))
    axs[0].set_xlabel('initial peak stress $\\sigma_{\\mathrm{ip}}$ (MPa)'); axs[0].set_ylabel('SEA (kJ kg$^{-1}$)')
    axs[0].set_position([0.085, 0.2, 0.30, 0.72]); axs[1].set_position([0.655, 0.2, 0.265, 0.72])
    cb = fig.colorbar(sc, cax=fig.add_axes([0.395, 0.2, 0.014, 0.72])); cb.set_label('$\\log_{10}\\sigma_{\\mathrm{plat}}$', labelpad=1)
    panel(axs[0], '(a) SEA against $\\sigma_{\\mathrm{ip}}$')
    axs[1].scatter(Y[:, 2], Y[:, 0], s=6, color='0.6', lw=0)
    sc2 = axs[1].scatter(Y[nd, 2], Y[nd, 0], s=22, c=Y[nd, 1], cmap='plasma', edgecolors='k', lw=0.6)
    cb2 = fig.colorbar(sc2, cax=fig.add_axes([0.93, 0.2, 0.014, 0.72])); cb2.set_label('$\\sigma_{\\mathrm{ip}}$ (MPa)', labelpad=1)
    axs[1].plot(np.log10(u['plat_var']), u['SEA'], '*', color='r', ms=11, mec='k')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
    axs[1].set_xlabel('$\\log_{10}\\sigma_{\\mathrm{plat}}$ (MPa$^2$)'); axs[1].set_ylabel('SEA (kJ kg$^{-1}$)')
    panel(axs[1], '(b) SEA against $\\sigma_{\\mathrm{plat}}$')
    fig.savefig(FIGDIR + 'fig15_pareto_front.pdf')
    return Y, nd


def figS2(R):
    fig, ax = plt.subplots(1, 1, figsize=(FW, 2.2))
    for a in [x for x in ('A', 'B', 'C', 'A0', 'B0', 'Bc') if x in R]:
        M = []
        for r in R[a]:
            lam = [o['lam'] for o in sorted(r['obs'], key=lambda o: o['t_end'])]
            c = np.cumsum(lam); ax.plot(np.arange(1, len(c) + 1), c, color=ARM_COL[a], lw=0.4, alpha=0.5); M.append(c)
        L = min(len(m) for m in M); ax.plot(np.arange(1, L + 1), np.median([m[:L] for m in M], 0), color=ARM_COL[a], lw=1.5, label=ARM_LABEL[a])
    ax.set_xlabel('evaluations returned'); ax.set_ylabel('cumulative HF evaluations'); ax.legend(frameon=False, ncol=2, loc='upper left', labelspacing=0.25)
    fig.savefig(FIGDIR + 'figS2_fidelity_allocation.pdf')


def figS3(R):
    arms = [a for a in ('A', 'B', 'C', 'A0', 'B0', 'Bc') if a in R]
    fig, axs = plt.subplots(2, 3, figsize=(FW, 3.8), sharex=True, sharey=True)
    for ax, a in zip(axs.ravel(), arms):
        for r in R[a]:
            t = np.array([x[0] for x in r['trace']]) / 3600; hv = np.maximum.accumulate([x[2] for x in r['trace']])
            o = np.argsort(t); ax.step(t[o], np.array(hv)[o], where='post', lw=0.8)
        ax.set_title(ARM_LABEL[a], fontsize=8, loc='left')
    for ax in axs[1]: ax.set_xlabel('wall-clock time (h)')
    for ax in axs[:, 0]: ax.set_ylabel('hypervolume')
    fig.subplots_adjust(hspace=0.35, wspace=0.12)
    fig.savefig(FIGDIR + 'figS3_per_seed_hypervolume.pdf')


def fig9(R, settings, seed=0):
    from campaign import fit_classifier, prob_feasible, SC
    from doe import normalise, cdens, TMIN, RHO
    r = [x for x in R['B'] if x['seed'] == seed][0]
    fe = [o for o in r['obs'] if o['status'] in ('completed', 'pruned')]
    X = np.array([SC.fwd(normalise(o['q'])) for o in fe]); lab = np.array([1.0 if o['feasible'] else 0.0 for o in fe])
    clf = fit_classifier(X, lab)
    g = np.linspace(0.3, 2.0, 70)
    G0, G3 = np.meshgrid(g, g, indexing='xy')
    Q = normalise(np.column_stack([G0.ravel(), np.ones(G0.size), np.ones(G0.size), G3.ravel()]))
    p = prob_feasible(clf, SC.fwd(Q)).reshape(G0.shape)
    tm = cdens().tmin_batch(Q, RHO).reshape(G0.shape)
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.8))
    for ax, Z, lab_, cm in ((axs[0], p, 'P(feasible)', 'RdYlGn'), (axs[1], p * (1 - p), 'p(1-p)', 'magma')):
        Zm = np.ma.masked_where(tm < TMIN, Z)
        im = ax.pcolormesh(G0, G3, Zm, cmap=cm, shading='auto', vmin=0, vmax=(1 if cm == 'RdYlGn' else 0.25))
        ax.contourf(G0, G3, (tm < TMIN).astype(float), levels=[0.5, 1.5], colors=['0.85'])
        fig.colorbar(im, ax=ax, label=lab_)
        ax.set_xlabel('$q_0$ (with $q_1=q_2=1$)'); ax.set_ylabel('$q_3$')
    # evaluated designs projected (only those close to the slice)
    for o in fe:
        qn = np.array(o['q']); qq = qn / (0.5 * (qn[1] + qn[2]))
        if abs(qq[1] - 1) < 0.2 and abs(qq[2] - 1) < 0.2 and 0.3 <= qq[0] <= 2.0 and 0.3 <= qq[3] <= 2.0:
            axs[0].plot(qq[0], qq[3], 'x' if o['status'] == 'pruned' else 'o', color='k', ms=4, mfc='none')
    for ax in axs:
        ax.set_xlim(0.3, 2.0); ax.set_ylim(0.3, 2.0)
    panel(axs[0], '(a) Posterior probability of feasibility')
    panel(axs[1], '(b) Classifier uncertainty')
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(FIGDIR + 'fig09_feasibility_classifier.pdf')


def fig_baselines(R):
    """Comparison with established approaches at equal CPU budget (Section 8.2.4)."""
    arms = [('B', 'Mode B (this work)'), ('A0c', 'synchronous MF-BO'), ('S', 'synchronous SF-BO'),
            ('Sb', 'asynchronous SF-BO'), ('R', 'space-filling search')]
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.55), gridspec_kw=dict(width_ratios=[1.35, 1]))
    tg = np.linspace(0, 2.7, 271)
    for a, lab in arms:
        H = np.array([hv_on_grid(r['trace'], tg * 3600, 'time') for r in R[a]])
        axs[0].plot(tg, np.median(H, 0), color=ARM_COL[a], label=lab, lw=1.3 if a == 'B' else 1.0)
        axs[0].fill_between(tg, np.percentile(H, 25, 0), np.percentile(H, 75, 0), color=ARM_COL[a], alpha=0.15, lw=0)
    axs[0].set_xlabel('wall-clock time (h)'); axs[0].set_ylabel('hypervolume (HF)')
    axs[0].set_xlim(0, 2.7); axs[0].set_ylim(0, None)
    axs[0].legend(frameon=False, loc='lower right', handlelength=1.4, labelspacing=0.25)
    panel(axs[0], '(a) Hypervolume against wall-clock time')
    ticks = ['B', 'MF\\\\sync', 'SF\\\\sync', 'SF\\\\async', 'space-\\\\filling']
    for i, (a, lab) in enumerate(arms):
        v = np.array([r['hv_final'] for r in R[a]])
        axs[1].boxplot([v], positions=[i], widths=0.5, showfliers=False, medianprops=dict(color='k'),
                       boxprops=dict(color=ARM_COL[a]), whiskerprops=dict(color=ARM_COL[a]), capprops=dict(color=ARM_COL[a]))
        axs[1].plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, 'o', ms=3, color=ARM_COL[a])
    axs[1].set_xticks(range(len(arms)))
    axs[1].set_xticklabels(['B', 'MF\nsync', 'SF\nsync', 'SF\nasync', 'space-\nfilling'])
    axs[1].set_ylabel('terminal hypervolume')
    panel(axs[1], '(b) Terminal hypervolume')
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(FIGDIR + 'fig14b_baseline_comparison.pdf')


if __name__ == '__main__':
    R = load_runs(); settings = json.load(open(DATA + 'settings.json'))
    for f in sys.argv[1:]:
        if f in ('fig15', 'fig9'):
            globals()[f](R, settings)
        else:
            globals()[f](R)
