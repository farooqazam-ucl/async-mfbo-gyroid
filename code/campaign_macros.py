"""Macros for Section 8.2 (campaign comparison) and 8.3 (confirmation)."""
import os as _os
ORIG_ARMS = ('A', 'B', 'C', 'A0', 'B0', 'Bc')  # arms of the execution-mode study
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import json, os
import numpy as np
DATA = _ROOT + 'data/'
S_LIMIT = json.load(open(DATA + 'settings.json'))['sigma_limit']


def add(M, setm, f, pct):
    C = json.load(open(DATA + 'campaign_stats.json'))
    S, T = C['stats'], C['tests']
    arms = [a for a in ('A', 'B', 'C', 'A0', 'B0', 'Bc') if a in S]

    def mi(a, k, nd=2, scale=1.0):
        s = S[a][k]
        if not np.isfinite(s['med']):
            return '$>T_{\\mathrm{wall}}$'
        return '%s [%s, %s]' % (f(s['med'] * scale, nd), f(s['q1'] * scale, nd), f(s['q3'] * scale, nd))

    def mis(a, k, nd=2, scale=1.0):
        """Median above, interquartile range below (stacked table cell)."""
        s = S[a][k]
        if not np.isfinite(s['med']):
            return '$>T_{\\mathrm{wall}}$'
        if abs(s['q1'] - s['q3']) * scale < 0.5 * 10 ** (-nd):
            return '\\begin{tabular}[t]{@{}c@{}}%s\\\\[-2pt]{\\scriptsize[%s]}\\end{tabular}' % (f(s['med'] * scale, nd), f(s['q1'] * scale, nd))
        return '\\begin{tabular}[t]{@{}c@{}}%s\\\\[-2pt]{\\scriptsize[%s, %s]}\\end{tabular}' % (f(s['med'] * scale, nd), f(s['q1'] * scale, nd),
                                                              f(s['q3'] * scale, nd))

    rows = []
    spec = [('Wall-clock time $T_{\\mathrm{wall}}$ (h)', 'T_wall', 2, 1.0),
            ('Time-to-target (h)', 'ttt', 2, 1.0),
            ('CPU time (h)', 'cpu_hours', 2, 1.0),
            ('Worker utilisation (\\%)', 'utilisation', 0, 100.0),
            ('Analyses pruned', 'n_pruned', 0, 1.0),
            ('CPU saved by pruning (h)', 'cpu_saved_h', 2, 1.0),
            ('Terminal hypervolume', 'hv_final', 2, 1.0),
            ('HF evaluations', 'n_hf', 0, 1.0),
            ('Decision time (s)', 'mean_decision_s', 2, 1.0)]
    for lab, k, nd, sc in spec:
        rows.append(lab + ' & ' + ' & '.join(mis(a, k, nd, sc) for a in arms) + '\\\\[2pt]')
    setm('campaigntablerows', '\n'.join(rows))
    setm('campaigntablehead', ' & '.join(arms))
    setm('campaigntablecols', 'l' + 'c' * len(arms))
    med = lambda a, k: S[a][k]['med']
    # H1
    for a, b, tag in (('A', 'B', ''), ('A0', 'B0', 'np')):
        if a in S and b in S:
            red = 100 * (1 - med(b, 'T_wall') / med(a, 'T_wall'))
            setm('TwallA' + tag, f(med(a, 'T_wall'))); setm('TwallB' + tag, f(med(b, 'T_wall')))
            setm('Twallred' + tag, pct(red, 0)); setm('speedup' + tag, f(med(a, 'T_wall') / med(b, 'T_wall')))
            setm('utilA' + tag, pct(100 * med(a, 'utilisation'), 0)); setm('utilB' + tag, pct(100 * med(b, 'utilisation'), 0))
            p = T['%s_vs_%s' % (a, b)]['T_wall']
            setm('pTwall' + tag, ('%.3f' % p) if p >= 0.001 else '<0.001')
            setm('cpuA' + tag, f(med(a, 'cpu_hours'))); setm('cpuB' + tag, f(med(b, 'cpu_hours')))
            setm('hvA' + tag, f(med(a, 'hv_final'))); setm('hvB' + tag, f(med(b, 'hv_final')))
            setm('pHV' + tag, '%.2f' % T['%s_vs_%s' % (a, b)]['hv_final'])
    if 'A' in S and 'B' in S:
        setm('tttA', f(med('A', 'ttt'))); setm('tttB', f(med('B', 'ttt')))
        setm('pTTT', '%.3f' % T['A_vs_B']['ttt'])
    # H2
    for a, b, tag in (('B', 'B0', 'B'), ('A', 'A0', 'A')):
        if a in S and b in S:
            setm('cpuPrune' + tag, f(med(a, 'cpu_hours'))); setm('cpuNoPrune' + tag, f(med(b, 'cpu_hours')))
            setm('cpuRed' + tag, pct(100 * (1 - med(a, 'cpu_hours') / med(b, 'cpu_hours')), 0))
            setm('pCPU' + tag, '%.3f' % T['%s_vs_%s' % (a, b)]['cpu_hours'])
            setm('hvPrune' + tag, f(med(a, 'hv_final'))); setm('hvNoPrune' + tag, f(med(b, 'hv_final')))
            setm('pHVprune' + tag, '%.2f' % T['%s_vs_%s' % (a, b)]['hv_final'])
            setm('nPruned' + tag, f(med(a, 'n_pruned'), 0))
            setm('cpuSaved' + tag, f(med(a, 'cpu_saved_h'))); setm('cpuSavedEst' + tag, f(med(a, 'cpu_saved_est_h')))
            setm('twallPrune' + tag, f(med(a, 'T_wall'))); setm('twallNoPrune' + tag, f(med(b, 'T_wall')))
            setm('pTwallPrune' + tag, '%.3f' % T['%s_vs_%s' % (a, b)]['T_wall'])
    if 'C' in S:
        setm('meanPoolC', f(med('C', 'mean_pool'), 1)); setm('nhfC', f(med('C', 'n_hf'), 0))
        setm('TwallC', f(med('C', 'T_wall'))); setm('cpuC', f(med('C', 'cpu_hours'))); setm('hvC', f(med('C', 'hv_final')))
        setm('utilC', pct(100 * med('C', 'utilisation'), 0))
        setm('TwallredC', pct(100 * (1 - med('C', 'T_wall') / med('B', 'T_wall')), 0))
        setm('pTwallBC', '%.3f' % T['B_vs_C']['T_wall']); setm('pHVBC', '%.2f' % T['B_vs_C']['hv_final'])
    # comparison with established approaches (Section 8.2.4)
    base = [a for a in ('B', 'A0c', 'S', 'Sb', 'R') if a in S]
    if len(base) == 5:
        rows = []
        for lab, k, nd, sc in [('Terminal hypervolume', 'hv_final', 2, 1.0),
                               ('Wall-clock time $T_{\\mathrm{wall}}$ (h)', 'T_wall', 2, 1.0),
                               ('CPU time (h)', 'cpu_hours', 2, 1.0),
                               ('Time-to-target (h)', 'ttt', 2, 1.0),
                               ('High-fidelity analyses', 'n_hf', 0, 1.0),
                               ('Low-fidelity analyses', 'n_lf', 0, 1.0)]:
            if k == 'n_lf':
                cells = []
                for a in base:
                    v = 100 - S[a]['n_hf']['med'] if a in ('B', 'A0c') else 0
                    cells.append(f(v, 0))
                rows.append(lab + ' & ' + ' & '.join(cells) + '\\\\')
                continue
            rows.append(lab + ' & ' + ' & '.join(mis(a, k, nd, sc) for a in base) + '\\\\[2pt]')
        setm('baselinetablerows', '\n'.join(rows))
        pv = lambda x: ('%.3f' % x) if x >= 0.001 else '<0.001'
        for b, tag in (('A0c', 'MFs'), ('S', 'SFs'), ('Sb', 'SFa'), ('R', 'Rnd')):
            setm('hv' + tag, f(med(b, 'hv_final'))); setm('Twall' + tag, f(med(b, 'T_wall')))
            setm('cpu' + tag, f(med(b, 'cpu_hours'))); setm('ttt' + tag, f(med(b, 'ttt')))
            setm('pHVB' + tag, pv(T['B_vs_%s' % b]['hv_final'])); setm('pTwallB' + tag, pv(T['B_vs_%s' % b]['T_wall']))
            setm('pTTTB' + tag, pv(T['B_vs_%s' % b]['ttt']))
            setm('hvGain' + tag, pct(100 * (med('B', 'hv_final') / med(b, 'hv_final') - 1), 0))
        setm('pHVSFsRnd', pv(T['S_vs_R']['hv_final'])); setm('pHVSFaRnd', pv(T['Sb_vs_R']['hv_final']))
        setm('pHVMFsSFs', pv(T['A0c_vs_S']['hv_final']))
        setm('hfBudgetSF', f(S['S']['n_hf']['med'], 0))
        setm('TwallredMFs', pct(100 * (1 - med('B', 'T_wall') / med('A0c', 'T_wall')), 0))
    if 'Bc' in S:
        setm('hvBc', f(med('Bc', 'hv_final'))); setm('tttBc', f(med('Bc', 'ttt')) if np.isfinite(med('Bc', 'ttt')) else 'not reached')
        setm('pHVBBc', '%.2f' % T['B_vs_Bc']['hv_final']); setm('pTTTBBc', '%.2f' % T['B_vs_Bc']['ttt'])
    # pruning details from the campaign logs
    import glob
    eps, frac = [], []
    for fn in sorted(glob.glob(DATA + 'campaigns/B_s*.json')) + sorted(glob.glob(DATA + 'campaigns/A_s*.json')):
        r = json.load(open(fn))
        P = [o for o in r['obs'] if o['status'] == 'pruned']
        eps += [o['eps_end'] for o in P]
        if P:
            frac.append(sum(o['full_duration'] - o['duration'] for o in P) / sum(o['full_duration'] for o in P))
    setm('pruneStrain', f(np.median(eps)))
    setm('pruneSaveFrac', pct(100 * np.median(frac), 0))
    setm('decB', f(med('B', 'mean_decision_s'), 1))
    # confirmation
    if os.path.exists(DATA + 'confirm_results.json'):
        CR = json.load(open(DATA + 'confirm_results.json'))
        lab = {'uniform': 'Uniform', 'maxSEA': 'Max.\\ SEA$^{\\ast}$', 'minIP': 'Min.\\ $\\sigma_{\\mathrm{ip}}^{\\ast}$',
               'minVAR': 'Min.\\ $\\sigma_{\\mathrm{plat}}^{\\ast}$', 'knee': 'Knee$^{\\ast}$'}
        rows = []
        for k in ('uniform', 'maxSEA', 'knee', 'minIP', 'minVAR'):
            if k not in CR:
                continue
            c = CR[k]; em = c.get('emulator')
            rows.append('%s & (%s) & %s & %s & %s & %s & %s\\\\' % (
                lab[k], ', '.join('%.2f' % v for v in c['q']),
                f(c['SEA']) + ((' (%s)' % f(em[0])) if em else ''),
                f(c['ip']) + ((' (%s)' % f(-em[1])) if em else ''),
                '%.3f' % c['plat_var'] + ((' (%.3f)' % (10 ** (-em[2]))) if em else ''),
                f(c['smax']), '%.2f' % c['tmin']))
        setm('confirmrows', '\n'.join(rows))
        u = CR['uniform']
        G = {k: c for k, c in CR.items() if k != 'uniform'}
        sg = [100 * (c['SEA'] / u['SEA'] - 1) for c in G.values()]
        ir = [100 * (1 - c['ip'] / u['ip']) for c in G.values()]
        vr = [c['plat_var'] / u['plat_var'] for c in G.values()]
        sm = [c['smax'] for c in G.values()]
        setm('confN', str(len(G)))
        setm('confSEAlo', pct(min(sg), 1)); setm('confSEAhi', pct(max(sg), 1))
        setm('confIPlo', pct(min(ir), 0)); setm('confIPhi', pct(max(ir), 0))
        setm('confVarlo', f(min(vr), 0)); setm('confVarhi', f(max(vr), 0))
        setm('confSmaxlo', f(min(sm))); setm('confSmaxhi', f(max(sm)))
        nv = sum(1 for c in G.values() if c['smax'] > S_LIMIT)
        words = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        setm('confNviol', words[nv])
        setm('confViolMax', pct(100 * (max(sm) / S_LIMIT - 1), 1))
        eS = [100 * abs(c['SEA'] - c['emulator'][0]) / c['SEA'] for c in G.values()]
        eI = [100 * abs(c['ip'] + c['emulator'][1]) / c['ip'] for c in G.values()]
        eV = [abs(np.log10(c['plat_var']) + c['emulator'][2]) for c in G.values()]
        setm('confErrSEA', pct(max(eS), 1)); setm('confErrIP', pct(max(eI), 0)); setm('confErrVar', f(max(eV), 1))
        er = [c for c in G.values() if c.get('eroded', 0) > 0]
        setm('confEroded', {0: 'None', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four'}.get(len(er), str(len(er))))

    # Pareto front statistics (emulator) and correlations
    import torch
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from scipy.stats import spearmanr
    obs = []
    for fn in sorted(glob.glob(DATA + 'campaigns/*.json')):
        r = json.load(open(fn))
        if r['arm'] not in ORIG_ARMS:
            continue
        obs += [o for o in r['obs'] if o['lam'] == 1 and o['status'] == 'completed' and o['feasible']]
    Y = np.array([o['y'] for o in obs]); nd = is_non_dominated(torch.as_tensor(Y)).numpy()
    P = Y[nd]
    setm('nFeasHF', str(len(obs))); setm('nND', str(int(nd.sum())))
    CD = json.load(open(DATA + 'confirm_designs.json'))
    setm('selNfeas', str(CD['n_feasible_hf'])); setm('selNnd', str(CD['n_pareto']))
    setm('frontSEAlo', f(P[:, 0].min())); setm('frontSEAhi', f(P[:, 0].max()))
    setm('frontIPlo', f(-P[:, 1].max())); setm('frontIPhi', f(-P[:, 1].min()))
    setm('frontVarlo', '%.3f' % (10 ** (-P[:, 2].max()))); setm('frontVarhi', '%.2f' % (10 ** (-P[:, 2].min())))
    Q = np.array([o['q'] for o in obs])[nd]; Qn = Q / Q.mean(1, keepdims=True)
    setm('spearIPqthree', '%.2f' % spearmanr(Qn[:, 3], -P[:, 1])[0])
    setm('spearLVqthree', '%.2f' % spearmanr(Qn[:, 3], -P[:, 2])[0])
    if os.path.exists(DATA + 'confirm_results.json'):
        CR = json.load(open(DATA + 'confirm_results.json'))
        if 'minIP' in CR:
            from geometry import ContinuumDensity, bezier_profile
            cdn = ContinuumDensity(); q = np.array(CR['minIP']['q']); s = cdn.scale(q, 0.25)
            uu = np.linspace(0, 1, 201); t = s * bezier_profile(uu, q)
            setm('minIPtmax', '%.1f' % t.max()); setm('minIPttop', '%.1f' % t[-1])
        nv = sum(1 for k, c in CR.items() if k != 'uniform' and c['smax'] > S_LIMIT)
        setm('confNviolCap', {0: 'None', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four'}[nv])
