"""Generate results_macros.tex: every number quoted in the manuscript is
computed here from the stored FE and campaign results."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, glob, os
import numpy as np
sys.path.insert(0, _ROOT + 'code')
from analyse_fe import record, load, RUNS
from geometry import make_design_mesh

DATA = _ROOT + 'data/'
RUNS = _ROOT + 'runs/'
# The LaTeX macros are written to output/results_macros.tex (override with GYROID_MACROS).
OUT = _os.environ.get('GYROID_MACROS', _ROOT + 'output/results_macros.tex')
_os.makedirs(_os.path.dirname(OUT), exist_ok=True)
M = {}


_SLAB_CACHE = None


def slab_profile(run, key, nb=10):
    """Mean equivalent plastic strain in `nb` slabs of the undeformed column height
    at nominal strain `key` (e.g. '0.100').  Computed from the snapshot file
    runs/<run>_snaps.npz when it is present (raw-data archive); otherwise read
    from data/slab_profiles.json, which stores the same numbers (written by
    code/make_slab_cache.py)."""
    global _SLAB_CACHE
    fn = RUNS + run + '_snaps.npz'
    if os.path.exists(fn):
        S = np.load(fn); eijk = S['eijk']; nz = eijk[:, 2].max() + 1
        idx = np.minimum(((eijk[:, 2] + 0.5) / nz * nb).astype(int), nb - 1); ep = S['ep_' + key]
        return np.array([ep[idx == i].mean() for i in range(nb)])
    if _SLAB_CACHE is None:
        _SLAB_CACHE = json.load(open(DATA + 'slab_profiles.json'))
    return np.array(_SLAB_CACHE['%s|%s|%d' % (run, key, nb)])


def f(x, nd=2):
    return ('%.' + str(nd) + 'f') % x


def pct(x, nd=1):
    return ('%.' + str(nd) + 'f') % x + '\\%'


def setm(name, val):
    import re
    if isinstance(val, str):
        val = re.sub(r'^-(?=\d)', r'\\ensuremath{-}', val)
    M[name] = val


def iqr(v):
    return np.percentile(v, 25), np.percentile(v, 75)


def fe_macros():
    S = json.load(open(DATA + 'settings.json'))
    R = {r['name']: r for r in json.load(open(DATA + 'fe_records.json'))}
    clean = json.load(open(DATA + 'clean_timing.json'))
    setm('sigzero', f(S['sig0'], 1)); setm('Qvoce', f(S['Qv'], 1)); setm('satstress', f(S['sig0'] + S['Qv'], 1))
    setm('costratio', f(S['cost_ratio'], 2)); setm('siglimit', f(S['sigma_limit'], 1))
    setm('seatarget', f(S['sea_target'], 1)); setm('kappazero', f(S['kappa0'], 3))
    setm('siglimitjust', S['siglimit_text']); setm('refpointtext', S['ref_text'])
    d, m = load(RUNS + 'doe/hf_00')
    setm('loadtime', f(m['T'] * 1e3, 2)); setm('platenspeed', f(0.6 * 100.0 / (m['T'] * 1e3) , 1))
    # timing
    lf = np.array([R['doe/' + k]['wall'] for k, c in clean.items() if c and k.startswith('lf')])
    hf = np.array([R['doe/' + k]['wall'] for k, c in clean.items() if c and k.startswith('hf')])
    setm('cleanLF', str(len(lf))); setm('cleanHF', str(len(hf)))
    setm('lfTime', f(np.median(lf), 0)); setm('hfTime', f(np.median(hf), 0))
    a, b = iqr(lf); setm('lfTimeIQR', '%s--%s' % (f(a, 0), f(b, 0)))
    a, b = iqr(hf); setm('hfTimeIQR', '%s--%s' % (f(a, 0), f(b, 0)))
    # campaign totals
    fe = [r for r in R.values()]
    setm('nFEtotal', str(len(fe)))
    doe = [r for r in fe if r['name'].startswith(('doe/', 'test/'))]
    setm('nDoE', str(len(doe)))
    setm('cpuFEtotal', f(sum(r['wall'] for r in fe) / 3600.0, 1))
    # DoE ranges at HF
    H = [r for r in fe if r['name'].startswith('doe/hf')]
    setm('doeSEAmin', f(min(r['SEA'] for r in H))); setm('doeSEAmax', f(max(r['SEA'] for r in H)))
    setm('doeIPmin', f(min(r['ip'] for r in H))); setm('doeIPmax', f(max(r['ip'] for r in H)))
    setm('doeLVmin', f(min(np.log10(r['plat_var']) for r in H))); setm('doeLVmax', f(max(np.log10(r['plat_var']) for r in H)))
    setm('doeSMmin', f(min(r['smax'] for r in H))); setm('doeSMmax', f(max(r['smax'] for r in H)))
    u = R['doe/hf_00']
    setm('uniSEA', f(u['SEA'])); setm('uniIP', f(u['ip'])); setm('uniSM', f(u['smax'])); setm('uniVar', f(u['plat_var'], 3))
    inf = [r for r in H if r['smax'] > S['sigma_limit']]
    words = {0: 'None', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
    setm('doeInfeasible', words.get(len(inf), str(len(inf))))
    ech = [r['eps_cross'] for r in H if np.isfinite(r.get('eps_cross', np.inf))]
    ecl = [r['eps_cross'] for r in fe if r['name'].startswith('doe/lf') and np.isfinite(r.get('eps_cross', np.inf))]
    setm('crossHFmin', f(min(ech))); setm('crossHFmax', f(max(ech)))
    setm('crossLFmin', f(min(ecl))); setm('crossLFmax', f(max(ecl)))
    # quasi-static and energy diagnostics over DoE + test
    ke02 = []
    for r in doe:
        dd, _ = load(RUNS + r['name']); e = dd['strain']; tt = e / e[-1]
        ke02.append(float((dd['KE'] / np.maximum(dd['IE'], 1e-12))[tt >= 0.2].max()))
    ke02 = np.array(ke02)
    setm('qsPass', str(int(np.sum(ke02 < 0.05)))); setm('qsN', str(len(ke02))); setm('qsMax', f(ke02.max(), 3))
    setm('hgLF', pct(100 * np.median([r['hg_ie'] for r in doe if r['lam'] == 0]), 1))
    setm('hgHF', pct(100 * np.median([r['hg_ie'] for r in doe if r['lam'] == 1]), 1))
    setm('addedMax', pct(100 * max(r['added_mass'] for r in doe), 1))
    er = R['doe/hf_12']
    setm('erosionfraction', pct(100 * er['eroded'] / er['ne'], 1))
    fr = [er['eroded'] / er['ne']]
    nconf = 0
    if os.path.exists(DATA + 'confirm_results.json'):
        for k, c in json.load(open(DATA + 'confirm_results.json')).items():
            if c.get('eroded', 0) > 0:
                fr.append(c['eroded'] / c.get('ne', 46656)); nconf += 1
    setm('erosionmax', pct(100 * max(fr), 1))
    setm('erosionConfirmN', {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}[nconf])
    setm('erosionTotalN', {1: 'One analysis', 2: 'Two analyses', 3: 'Three analyses', 4: 'Four analyses', 5: 'Five analyses'}[1 + nconf])
    # rate study
    rate = {16: R['doe/lf_00']}
    for tf in (24, 32):
        if 'rate/u_N24_Tf%d' % tf in R:
            rate[tf] = R['rate/u_N24_Tf%d' % tf]
    if len(rate) == 3:
        dS = 100 * (rate[16]['SEA'] - rate[32]['SEA']) / rate[32]['SEA']
        dI = 100 * (rate[16]['ip'] - rate[32]['ip']) / rate[32]['ip']
        dS24 = 100 * (rate[24]['SEA'] - rate[32]['SEA']) / rate[32]['SEA']
        txt = ('Doubling the loading time (from $16T_1$ to $32T_1$, uniform design at low fidelity) reduced SEA by '
               '%s and the initial peak stress by %s, and the change from $24T_1$ to $32T_1$ was %s in SEA.  The '
               'production loading time therefore overestimates the quasi-static response by about %s; because every '
               'design is analysed with the same loading time, this inertial contribution largely cancels in the '
               'comparisons between designs.' % (
                   pct(dS, 1), pct(dI, 1), pct(dS24, 1), pct(round(dS), 0)))
    else:
        txt = ''
    setm('quasistaticsentence', txt)
    return R, S


def conv_macros(R):
    conv = {16: 'conv/u_N16', 24: 'doe/lf_00', 30: 'conv/u_N30', 36: 'doe/hf_00', 42: 'conv/u_N42', 48: 'conv/u_N48'}
    Ns = [N for N in [16, 24, 30, 36, 48] if conv[N] in R]
    rows = []
    for N in [16, 24, 30, 36, 48]:
        mesh, _ = make_design_mesh([1, 1, 1, 1], 0.25, 25.0, N, 4)
        use = {24: 'LF ($\\lambda=0$)', 36: 'HF ($\\lambda=1$)', 48: 'reference'}.get(N, 'convergence')
        rows.append('%d & $%d^3$ & %.3f & %.2f & %s & %.4f & %s\\\\' % (
            N, 4 * N, 25.0 / N, 2.09 * N / 25.0, '{:,}'.format(len(mesh['elems'])).replace(',', '\\,'), mesh['rho'], use))
    setm('resolutionrows', '\n    '.join(rows))
    ref = R[conv[48]]
    rows = []
    for N in Ns:
        r = R[conv[N]]
        e = lambda k: abs(r[k] - ref[k]) / abs(ref[k]) * 100
        if N == 48:
            rows.append('%d & %.2f & %.2f & %.2f & %.4f & %.3f\\\\' % (N, 2.09 * N / 25, r['SEA'], r['ip'], r['plat_var'], r['hg_ie']))
        else:
            rows.append('%d & %.2f & %.2f (%.1f\\%%) & %.2f (%.1f\\%%) & %.4f & %.3f\\\\' % (
                N, 2.09 * N / 25, r['SEA'], e('SEA'), r['ip'], e('ip'), r['plat_var'], r['hg_ie']))
    setm('convrows', '\n    '.join(rows))
    h = R[conv[36]]; l = R[conv[24]]
    rel = lambda r, k: 100 * (r[k] - ref[k]) / ref[k]
    def spct(x, nd=1):
        return ('$-$' if x < 0 else '+') + ('%.' + str(nd) + 'f') % abs(x) + '\\%'
    eS, eI = rel(h, 'SEA'), rel(h, 'ip'); lS, lI = rel(l, 'SEA'), rel(l, 'ip')
    fine = [abs(rel(R[conv[N]], k)) for N in (30, 36) for k in ('SEA', 'ip')]
    pv = [R[conv[N]]['plat_var'] for N in (24, 30, 36, 48)]
    dpv = max(abs(R[conv[N]]['plat_var'] - ref['plat_var']) for N in (24, 30, 36))
    setm('hfConvMax', pct(max(abs(eS), abs(eI)), 1))
    nest = []
    for k, r in R.items():
        if k.startswith('doe/hf_'):
            for k2, r2 in R.items():
                if k2.startswith('doe/lf_') and np.allclose(r2['q'], r['q']):
                    nest.append((r2, r))
    bias = [rel(l, 'SEA'), rel(l, 'smax')] + [100 * np.mean([(a[k] - b[k]) / b[k] for a, b in nest]) for k in ('SEA', 'smax')]
    setm('lfBiasLo', pct(min(bias), 0)); setm('lfBiasHi', pct(max(bias), 0))
    setm('lfBiasIP', pct(100 * np.mean([(a['ip'] - b['ip']) / b['ip'] for a, b in nest]), 0))
    txt = ('The responses converge as the mesh is refined, but not monotonically '
           '(Table~\\ref{tab:mesh_convergence} and Fig.~\\ref{fig:mesh_convergence}).  Coarse meshes are too stiff '
           'and too strong because a wall represented by one or two voxels bends mainly through the hourglass '
           'stabilisation: $N=16$ overestimates SEA by %s and $N=24$ by %s.  From $N=30$ on, SEA and '
           '$\\sigma_{\\mathrm{ip}}$ lie within %s of the $N=48$ values without a consistent trend; these residual '
           'differences reflect the discrete change of the voxel set along the curved wall, the mechanism that also '
           'makes the elastic modulus oscillate (Section~3.5.3).  The high-fidelity mesh deviates from $N=48$ by %s '
           'in SEA and %s in the initial peak stress, and the low-fidelity mesh by %s and %s.  The plateau variance '
           'of the uniform design is small ($\\sigma_{\\mathrm{plat}}\\le%s$~MPa$^2$ for $N\\ge24$), so its '
           'relative deviation is not meaningful; the absolute differences from $N=48$ are below %s~MPa$^2$.  The '
           'tolerance adopted for the high-fidelity level is 5\\%% on SEA and $\\sigma_{\\mathrm{ip}}$ relative '
           'to the finest affordable mesh; $N=36$ meets it while costing about a quarter of the CPU time of an '
           '$N=48$ analysis, which justifies its designation as the reference fidelity of the optimisation.  The '
           'low-fidelity mesh overestimates SEA and the maximum stress, a bias that is largely systematic and is '
           'absorbed by the inter-fidelity correlation of the surrogate (Section~8.1).  The voxel model with three '
           'to four elements through the wall is consistent with the guidance of Guldberg et al.\\ '
           '\\cite{guldberg1998}, who found that three to four voxel elements through a section are needed to avoid '
           'excessive stiffening in bending.') % (
        spct(rel(R[conv[16]], 'SEA'), 0).lstrip('+'), spct(lS).lstrip('+'), pct(max(fine), 1), spct(eS), spct(eI),
        spct(lS), spct(lI), '%.3f' % max(pv), '%.3f' % (np.ceil(dpv * 200) / 200))
    setm('convparagraph', txt)
    return ref


def elastic_macros():
    E = json.load(open(DATA + 'elastic.json'))
    conv = E['conv']
    Es = np.array([c['E_adj'] for c in conv]); Ns = np.array([c['N'] for c in conv])
    s = Ns >= 30
    mean = Es[s].mean(); lo = Es[s].min(); hi = Es[s].max()
    exp25 = 892 * 0.25 ** 1.39
    d = {round(x['rho'], 2): x for x in E['density']}
    fine = {round(x['rho'], 2): x for x in E.get('density_fine', [])}
    def dev(x):
        return 100 * (x['E'] - 892 * x['rho'] ** 1.39) / (892 * x['rho'] ** 1.39)
    txt = ('The effective modulus does not converge monotonically but oscillates as the resolution changes '
           '(Fig.~\\ref{fig:voxel_vs_conforming}a), the behaviour reported by Guldberg et al.\\ \\cite{guldberg1998} '
           'for voxel models of curved boundaries: the set of voxels retained along the wall changes discretely with '
           '$h$.  For $N\\ge30$ all values lie within %s--%s~MPa (mean %s~MPa, spread $\\pm%s$), whereas the two '
           'coarsest grids deviate by up to %s.  The stair-stepping error of the high-fidelity mesh in the elastic '
           'regime is therefore bounded by about %s, and of the low-fidelity mesh by about %s.  The modulus of the '
           'high-fidelity model is %s above the experimental power law at $\\bar\\rho=0.25$, %s above at 0.305 and '
           '%s above at 0.14 (Fig.~\\ref{fig:voxel_vs_conforming}b); refining the mesh to $N=56$ at $\\bar\\rho=0.14$ '
           'changes the modulus by only %s, so the larger deviation at low density is not a discretisation error.  '
           'It is consistent with the manufacturing defects of thin sintered walls --- porosity, surface roughness and '
           'partially fused particles reduce their effective thickness --- which the idealised geometry omits.') % (
        f(lo, 1), f(hi, 1), f(mean, 1), pct(100 * (hi - lo) / 2 / mean, 1),
        pct(100 * np.max(np.abs(Es[~s] - mean)) / mean, 0),
        pct(100 * max(abs(Es[Ns == 36][0] - mean), (hi - lo) / 2) / mean, 0),
        pct(100 * abs(Es[Ns == 24][0] - mean) / mean + 100 * (hi - lo) / 2 / mean, 0),
        pct(dev(d[0.25]), 0), pct(dev(d[0.31] if 0.31 in d else d[0.3]), 0), pct(dev(d[0.14]), 0),
        pct(100 * abs(fine[0.14]['E'] - d[0.14]['E']) / d[0.14]['E'], 1) if 0.14 in fine else 'little')
    setm('elasticparagraph', txt)


def validation_macros(R):
    V = {}
    for name, rho in (('val/r305_N36', 0.305), ('doe/hf_00', 0.25), ('val/r140_N48', 0.14)):
        if name in R:
            dd, m = load(RUNS + name)
            S = float(np.max(dd['stress'][dd['strain'] <= 0.3]))
            V[rho] = dict(S=S, Sexp=54 * m['rho'] ** 1.63, rho=m['rho'])
    if len(V) < 3:
        setm('validationparagraph', '')
        return V
    e = {k: 100 * (v['S'] - v['Sexp']) / v['Sexp'] for k, v in V.items()}

    def prof(r, key, nb=10):
        return slab_profile(r, key, nb)
    keys = ('0.100', '0.200', '0.300', '0.400', '0.500', '0.600')
    loc = {r: max((lambda p: p.max() / np.median(p))(prof(r, k)) for k in keys) for r in ('val/r305_N36',)}
    p2 = prof('val/r140_N48', '0.200'); loc14 = p2.max() / np.median(p2)
    setm('valErrTwentyFive', pct(abs(e[0.25]), 0))
    txt = ('After calibration the high-fidelity model reproduces the measured strength at $\\bar\\rho=0.305$ to within '
           '%s (%s against %s~MPa); the residual arises because $\\sigma_0$ and $Q$ were rescaled in proportion '
           'to the strength error of a preliminary analysis, whereas the strength is not exactly proportional to '
           'them.  Without further adjustment it predicts %s~MPa at the design density '
           '$\\bar\\rho=0.25$, %s from the experimental power law (%s~MPa), and %s~MPa at $\\bar\\rho=0.14$, %s from '
           'the measured %s~MPa.  The simulated strength therefore scales less steeply with density than the measured '
           'one, consistent with the defect sensitivity of thin walls noted above, and the model is most reliable at '
           'and above the design density.  The predicted deformation modes agree with the experiments '
           '(Fig.~\\ref{fig:experimental_validation}c,d).  At $\\bar\\rho=0.305$ the plastic strain is '
           'distributed evenly over the height: at no strain does the most strained tenth of the column carry more '
           'than %s times the median plastic strain.  At $\\bar\\rho=0.14$ it localises, first in the layers '
           'adjacent to the platens (%s times the median at $\\varepsilon=0.2$) and then in successive layers '
           'towards mid-height, the layer-by-layer mode that Abueidda et al.\\ \\cite{abueidda2019} observed at 7 '
           'and 14\\%% density.  The simulated response at $\\bar\\rho=0.14$ shows only shallow load drops '
           'between these collapses, whereas the measured curves oscillate strongly; the model contains no damage '
           'or fracture, which would deepen the drops.') % (
        pct(abs(e[0.305]), 1), f(V[0.305]['S']), f(V[0.305]['Sexp']), f(V[0.25]['S']), ('+' if e[0.25] > 0 else '') + pct(e[0.25], 1),
        f(V[0.25]['Sexp']), f(V[0.14]['S']), ('+' if e[0.14] > 0 else '') + pct(e[0.14], 0), f(V[0.14]['Sexp']),
        '%.2f' % loc['val/r305_N36'], '%.1f' % loc14)
    setm('validationparagraph', txt)
    return V


def parity_macros():
    P = json.load(open(DATA + 'parity.json'))
    mm = P['metrics']
    setm('parSEAhf', f(mm['SEA_HF']['R2'])); setm('parIPhf', f(mm['ip_HF']['R2'])); setm('parLVhf', f(mm['logvar_HF']['R2']))
    setm('rmseSEAhf', f(mm['SEA_HF']['RMSE'])); setm('rmseIPhf', f(mm['ip_HF']['RMSE'])); setm('rmseLVhf', f(mm['logvar_HF']['RMSE']))
    cov = np.mean([mm['%s_HF' % k]['cover95'] for k in ('SEA', 'ip', 'logvar')])
    setm('coverHF', pct(100 * cov, 0))
    if os.path.exists(DATA + 'parity_seeds.json'):
        PS = json.load(open(DATA + 'parity_seeds.json'))
        for key, tag in (('SEA_HF', 'SEA'), ('ip_HF', 'IP'), ('logvar_HF', 'LV')):
            v = np.array([PS[k][key] for k in PS])
            setm('par%srange' % tag, f(v.min()) + '--' + f(v.max())); setm('par%smed' % tag, f(np.median(v)))
        setm('parNseeds', {10: 'ten'}.get(len(PS), str(len(PS))))
    tc = P['task_corr']
    keys = list(tc.keys())
    def corr(i, j):
        return ', '.join(f(np.array(tc[k])[i, j]) if not isinstance(tc[k], str) else 'n/a' for k in keys)
    setm('taskcorrLFHF', corr(1, 2)); setm('taskcorrPHF', corr(0, 2))
    def cmean(i, j):
        v = [np.array(tc[k])[i, j] for k in keys if not isinstance(tc[k], str)]
        return f(min(v)) + '--' + f(max(v))
    setm('taskcorrLFHFshort', cmean(1, 2)); setm('taskcorrPHFshort', cmean(0, 2))
    R = {r['name']: r for r in json.load(open(DATA + 'fe_records.json'))}
    th = [r for k, r in R.items() if k.startswith('test/hf')]
    sdS = np.std([r['SEA'] for r in th]); sdI = np.std([r['ip'] for r in th])
    txt = ('The surrogate therefore captures the trend of the plateau variance reliably but ranks designs by SEA and '
           '$\\sigma_{\\mathrm{ip}}$ only coarsely: over the test designs these outputs have standard deviations of '
           '%s~kJ~kg$^{-1}$ and %s~MPa, comparable to the prediction errors.  This is one reason why the designs it '
           'selected were re-analysed with the FE model (Section~8.3).') % (f(sdS), f(sdI))
    setm('parityComment', txt)


PLACEHOLDERS = ['convrows', 'convparagraph', 'validationparagraph', 'valErrTwentyFive', 'parSEAhf', 'parIPhf', 'parLVhf',
                'rmseSEAhf', 'rmseIPhf', 'rmseLVhf', 'coverHF', 'taskcorrLFHF', 'taskcorrPHF', 'taskcorrLFHFshort',
                'taskcorrPHFshort', 'parityComment', 'quasistaticsentence', 'resolutionrows']


def collapse_macros():
    """Localisation of plastic strain over the height (Section 8.3, Fig. 16d)."""
    CR = json.load(open(DATA + 'confirm_results.json'))
    runs = {'uniform': 'doe/hf_00'}
    runs.update({k: c['run'] for k, c in CR.items() if k != 'uniform'})

    def prof(r, key, nb=10):
        return slab_profile(r, key, nb)
    names = {'uniform': 'locUni', 'maxSEA': 'locSEA', 'knee': 'locKnee', 'minIP': 'locIP', 'minVAR': 'locVAR'}
    for k, r in runs.items():
        p = prof(r, '0.100'); setm(names[k], '%.1f' % (p.max() / np.median(p)))
    p = prof(runs['minIP'], '0.200')
    setm('minIPtopTwo', '%.2f' % p[-1]); setm('minIPrestTwo', '%.2f' % p[:-1].max())
    sp = [(lambda p: (p.max() - p.min()) / p.mean())(prof(runs['uniform'], kk)) for kk in ('0.100', '0.200', '0.300', '0.400', '0.500')]
    setm('uniSpread', pct(100 * max(sp), 0))
    med = []
    for k, c in CR.items():
        if k != 'uniform' and c.get('eroded', 0) > 0:
            d = np.load(RUNS + c['run'] + '_hist.npz'); n = d['ninv']
            med.append(float(d['strain'][np.argmax(n >= 0.5 * n.max())]))
    if med:
        setm('delMedStrain', '%.2f' % min(med))


def write():
    missing = [k for k in PLACEHOLDERS if k not in M]
    for k in missing:
        M[k] = '\\textbf{[pending: %s]}' % k + ('\\\\' if k.endswith('rows') else '')
    if missing:
        print('PENDING MACROS:', missing)
    with open(OUT, 'w') as fh:
        fh.write('%% Generated by make_macros.py from the stored results. Do not edit by hand.\n')
        for k, v in M.items():
            fh.write('\\newcommand{\\%s}{%s}\n' % (k, v))


if __name__ == '__main__':
    R, S = fe_macros()
    try:
        conv_macros(R)
    except KeyError as ex:
        print('conv missing', ex)
    elastic_macros()
    validation_macros(R)
    if os.path.exists(DATA + 'parity.json'):
        parity_macros()
    if os.path.exists(DATA + 'confirm_results.json'):
        collapse_macros()
    if os.path.exists(_ROOT + 'code/campaign_macros.py'):
        import campaign_macros
        campaign_macros.add(M, setm, f, pct)
    write()
    print(len(M), 'macros written')
