import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, time
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, _ROOT + 'code')
from elastic import homogenise

out = {'conv': [], 'density': []}
for N in [16, 20, 24, 30, 36, 42, 48, 56]:
    t0 = time.time(); r = homogenise(N=N, rho=0.25)
    r['N'] = N; r['h'] = 25.0 / N
    r['E_adj'] = r['E'] * (0.25 / r['rho']) ** 1.39   # small density correction to exactly 0.25
    out['conv'].append({k: float(v) for k, v in r.items()})
    print('N', N, r, '%.0fs' % (time.time() - t0), flush=True)
    json.dump(out, open(_ROOT + 'data/elastic.json', 'w'), indent=1)
h = np.array([c['h'] for c in out['conv']]); E = np.array([c['E_adj'] for c in out['conv']])
f = lambda h, Einf, Cc, p: Einf + Cc * h ** p
popt, _ = curve_fit(f, h, E, p0=[E[-1], 10.0, 1.0], maxfev=20000)
out['extrap'] = dict(Einf=float(popt[0]), C=float(popt[1]), p=float(popt[2]))
print('extrap', out['extrap'], flush=True)
for rho in [0.10, 0.14, 0.20, 0.25, 0.305, 0.40]:
    r = homogenise(N=36, rho=rho)
    out['density'].append({k: float(v) for k, v in r.items()})
    print('rho', rho, r, flush=True)
    json.dump(out, open(_ROOT + 'data/elastic.json', 'w'), indent=1)
json.dump(out, open(_ROOT + 'data/elastic.json', 'w'), indent=1)
print('done')
