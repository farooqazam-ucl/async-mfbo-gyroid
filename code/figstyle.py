import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.figure

C = dict(A='#c0392b', B='#1f6fb2', C='#2e8b57', A0='#e59866', B0='#85c1e9', Bc='#7d3c98',
         A0c='#a04000', S='#566573', Sb='#17a589', R='#b7950b',
         lf='#e67e22', hf='#1f6fb2', grey='#7f8c8d', uni='#444444', exp='#000000')

# Figures are drawn at the text width of the manuscript (elsarticle preprint
# layout, 390 pt = 5.40 in) and included at \textwidth, so font sizes below are
# the sizes that appear in print.  All text is typeset by LaTeX in Latin Modern.
FW = 5.40
MINFS = 8.0
PREAMBLE = (r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}'
            r'\usepackage{amssymb}\usepackage{bm}')


def setup():
    plt.rcParams.update({
        'text.usetex': True, 'text.latex.preamble': PREAMBLE,
        'font.family': 'serif', 'font.serif': ['Latin Modern Roman'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'legend.fontsize': 8, 'legend.title_fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
        'lines.linewidth': 1.2, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'axes.titlepad': 4.0, 'figure.subplot.left': 0.09, 'figure.subplot.right': 0.985, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.3})


def enforce_min_fontsize(fig, minfs=MINFS):
    """Raise every piece of text in the figure to at least `minfs` points."""
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_fontsize() < minfs:
            t.set_fontsize(minfs)


_savefig = matplotlib.figure.Figure.savefig


def _savefig_checked(self, *args, **kwargs):
    enforce_min_fontsize(self)
    return _savefig(self, *args, **kwargs)


matplotlib.figure.Figure.savefig = _savefig_checked


def panel(ax, s):
    ax.set_title(r'\textbf{' + s + '}', loc='left', fontsize=9)


# Figures are written to output/figures/ (override with GYROID_FIGDIR).
FIGDIR = _os.environ.get('GYROID_FIGDIR', _ROOT + 'output/figures/')
FIGDIR = FIGDIR if FIGDIR.endswith('/') else FIGDIR + '/'
_os.makedirs(FIGDIR, exist_ok=True)
