import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json
sys.path.insert(0, _ROOT + 'code')
from elastic import homogenise
E = json.load(open(_ROOT + 'data/elastic.json'))
E['density_fine'] = []
for rho, N in [(0.14, 56), (0.10, 64), (0.20, 48)]:
    r = homogenise(N=N, rho=rho); r['N'] = N
    E['density_fine'].append({k: float(v) for k, v in r.items()})
    print(rho, N, r, flush=True)
    json.dump(E, open(_ROOT + 'data/elastic.json', 'w'), indent=1)
