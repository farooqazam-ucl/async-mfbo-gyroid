"""Figure 10: orchestrator schematic (drawn in inches at the text width)."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys
sys.path.insert(0, _ROOT + 'code')
from figstyle import setup, plt, C, FIGDIR, FW
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
setup()

H = 4.3  # figure height (in)


def box(ax, x, y, w, h, text, fc, ec='k', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.05', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, col='k', lw=0.9, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=col, lw=lw,
                                 shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(FW, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, H); ax.axis('off')

# ---------------- master process
ax.add_patch(FancyBboxPatch((0.05, 2.38), FW - 0.1, 1.86, boxstyle='round,pad=0.01,rounding_size=0.06',
                            fc='#f4f6f8', ec='0.45', lw=0.8, ls='--'))
ax.text(FW / 2, 4.08, r'\textbf{MASTER PROCESS}', ha='center', va='center', fontsize=8.5)
bw, gap, x0 = 1.6, 0.15, 0.15
box(ax, x0, 3.2, bw, 0.72, 'Analytical prior\nGibson--Ashby layers\nin series (zero cost)', '#eeeeee')
box(ax, x0 + bw + gap, 3.2, bw, 0.72, 'Multi-task GPs\nSEA, $\\sigma_{\\mathrm{ip}}$, $\\log_{10}\\sigma_{\\mathrm{plat}}$\n'
    'tasks: prior, LF, HF', '#dbe9f6')
box(ax, x0 + 2 * (bw + gap), 3.2, bw, 0.72, 'Feasibility classifier\nGP, Bernoulli likelihood\ncompleted and pruned', '#fde9d9')
arrow(ax, x0 + bw + 0.02, 3.56, x0 + bw + gap - 0.02, 3.56)
box(ax, x0, 2.5, 3 * bw + 2 * gap, 0.55,
    'Acquisition: qLogNEHVI on HF outputs $\\times$ $P(\\mathrm{feasible})$; running analyses are pending points\n'
    'screen and confirm: low-fidelity screens, one high-fidelity confirmation in six dispatches', '#e2f0d9')
for xc in (x0 + 1.5 * bw + gap, x0 + 2.5 * bw + 2 * gap):
    arrow(ax, xc, 3.18, xc, 3.07)

# ---------------- messages between master and workers
arrow(ax, 1.0, 2.36, 1.0, 1.72, col=C['B'])
ax.text(1.08, 2.04, 'dispatch $(\\boldsymbol\\theta,\\lambda)$', color=C['B'], ha='left', va='center', fontsize=8)
arrow(ax, 2.6, 1.72, 2.6, 2.36, col=C['C'])
ax.text(2.68, 2.04, 'completed:\nregression data', color=C['C'], ha='left', va='center', fontsize=8, linespacing=1.15)
arrow(ax, 4.9, 1.72, 4.9, 2.36, col=C['A'])
ax.text(4.82, 2.04, 'pruned:\nclassifier only', color=C['A'], ha='right', va='center', fontsize=8, linespacing=1.15)

# ---------------- worker pool
ax.add_patch(FancyBboxPatch((0.05, 0.3), FW - 0.1, 1.4, boxstyle='round,pad=0.01,rounding_size=0.06',
                            fc='white', ec='0.45', lw=0.8, ls='--'))
ax.text(FW / 2, 1.56, r'\textbf{WORKER POOL} ($W$ processes)', ha='center', va='center', fontsize=8.5)
cw, cg = 1.5, 0.27
cx = [0.15, 0.15 + cw + cg, FW - 0.15 - cw]
for k, xk in enumerate(cx):
    lab = ('W%d' % (k + 1)) if k < 2 else 'W$_W$'
    ax.text(xk + cw / 2, 1.36, lab, ha='center', va='center', fontsize=8)
    box(ax, xk, 1.0, cw, 0.24, 'voxel mesh, $N=24$ or 36', '#ffffff')
    box(ax, xk, 0.7, cw, 0.2, 'explicit solver', '#dbe9f6')
    box(ax, xk, 0.36, cw, 0.25, 'monitor: stop if 5-sample\nmean $>\\sigma_{\\mathrm{limit}}$', '#fde9d9')
    arrow(ax, xk + cw / 2, 0.985, xk + cw / 2, 0.915, ms=7)
    arrow(ax, xk + cw / 2 - 0.15, 0.685, xk + cw / 2 - 0.15, 0.625, ms=6)
    arrow(ax, xk + cw / 2 + 0.15, 0.625, xk + cw / 2 + 0.15, 0.685, col=C['A'], ms=6)
ax.text((cx[1] + cw + cx[2]) / 2, 0.8, '$\\cdots$', ha='center', va='center', fontsize=10)
ax.text(FW / 2, 0.14, 'No synchronisation barrier: a returning worker is refilled after one model update.',
        ha='center', va='center', fontsize=8, color='0.3')
fig.savefig(FIGDIR + 'fig10_framework_schematic.pdf')
