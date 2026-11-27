"""Figure 11: surrogate qualification on the independent test set."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json
import numpy as np
import torch
sys.path.insert(0, _ROOT + 'code')
from figstyle import setup, panel, plt, C, FIGDIR, FW
from campaign import build_mtgp, SC, to_max_form, TASK_PRIOR, TASK_LF, TASK_HF, load_pool
from run_campaigns import prior_data
setup()
DATA = _ROOT + 'data/'


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main(seed=None, save=True):
    """Fit the MTGPs with random seed `seed` (default: environment variable PARITY_SEED,
    else 0) and evaluate them on the test set.  With save=True, write Fig. 11 and
    data/parity.json.  Returns the accuracy metrics."""
    _sd = int(_os.environ.get('PARITY_SEED', '0')) if seed is None else int(seed)
    torch.manual_seed(_sd); np.random.seed(_sd)  # reproducible fits
    recs = json.load(open(DATA + 'fe_records.json'))
    tr = [r for r in recs if r['name'].startswith('doe/')]
    te = [r for r in recs if r['name'].startswith('test/')]
    pool, _ = load_pool(); prior = prior_data(pool)
    def XY(R):
        X = SC.fwd(np.array([np.array(r['q']) / np.mean(r['q']) for r in R]))
        T = np.array([TASK_HF if r['lam'] == 1 else TASK_LF for r in R])
        Y = to_max_form([r['SEA'] for r in R], [r['ip'] for r in R], [np.log10(max(r['plat_var'], 1e-4)) for r in R])
        return X, T, Y
    Xtr, Ttr, Ytr = XY(tr); Xte, Tte, Yte = XY(te)
    X = np.vstack([prior['X'], Xtr]); T = np.concatenate([np.full(len(prior['X']), TASK_PRIOR), Ttr]); Y = np.vstack([prior['Y'], Ytr])
    names = ['SEA (kJ kg$^{-1}$)', '$\\sigma_{\\mathrm{ip}}$ (MPa)', '$\\log_{10}\\sigma_{\\mathrm{plat}}$']
    sign = [1, -1, -1]
    out = {'metrics': {}, 'task_corr': {}}
    fig, axs = plt.subplots(1, 3, figsize=(FW, 2.4))
    for j in range(3):
        m = build_mtgp(X, T, Y[:, j], fit=True)
        # task correlation matrix
        try:
            Bk = m.covar_module.kernels[1]
            B = Bk.covar_matrix.to_dense().detach().numpy()
            d = np.sqrt(np.diag(B)); out['task_corr'][names[j]] = (B / np.outer(d, d)).tolist()
        except Exception as ex:
            out['task_corr'][names[j]] = str(ex)
        for lam, col, lab in ((1, C['hf'], 'HF'), (0, C['lf'], 'LF')):
            sel = Tte == (TASK_HF if lam == 1 else TASK_LF)
            Xq = torch.as_tensor(np.column_stack([Xte[sel], np.full(sel.sum(), TASK_HF if lam == 1 else TASK_LF)]))
            with torch.no_grad():
                post = m.posterior(Xq)
                mu = post.mean.numpy().ravel(); sd = post.variance.sqrt().numpy().ravel()
            yt = Yte[sel, j]
            yv, mv = sign[j] * yt, sign[j] * mu
            axs[j].errorbar(yv, mv, yerr=2 * sd, fmt='o' if lam == 1 else 's', color=col, ms=3.5, lw=0.7, capsize=1.5,
                            label='%s: $R^2=%.2f$' % (lab, r2(yt, mu)))
            out['metrics']['%s_%s' % (['SEA', 'ip', 'logvar'][j], lab)] = dict(
                R2=float(r2(yt, mu)), RMSE=float(np.sqrt(np.mean((yt - mu) ** 2))),
                cover95=float(np.mean(np.abs(yt - mu) <= 2 * sd)), n=int(sel.sum()))
        lo = min(axs[j].get_xlim()[0], axs[j].get_ylim()[0]); hi = max(axs[j].get_xlim()[1], axs[j].get_ylim()[1])
        axs[j].plot([lo, hi], [lo, hi], 'k--', lw=0.7); axs[j].set_xlim(lo, hi); axs[j].set_ylim(lo, hi)
        axs[j].set_xlabel('FE: ' + names[j]); axs[j].set_ylabel('MTGP prediction'); axs[j].legend(frameon=False, loc='lower right', handlelength=1.0, borderaxespad=0.1)
        axs[j].set_aspect('equal')
    fig.subplots_adjust(wspace=0.45)
    if save:
        fig.savefig(FIGDIR + 'fig11_surrogate_parity.pdf')
        json.dump(out, open(DATA + 'parity.json', 'w'), indent=1)
        print(json.dumps(out['metrics'], indent=0))
    plt.close(fig)
    return out['metrics']


if __name__ == '__main__':
    main()
