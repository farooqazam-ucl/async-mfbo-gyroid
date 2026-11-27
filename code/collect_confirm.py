import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, os, numpy as np
sys.path.insert(0, _ROOT + 'code')
from analyse_fe import record
from geometry import ContinuumDensity
DATA = _ROOT + 'data/'; RUNS = _ROOT + 'runs/'
cd = ContinuumDensity()
sel = json.load(open(DATA + 'confirm_designs.json'))['selected']
out = {}
u = record(RUNS + 'doe/hf_00')
out['uniform'] = dict(q=[1, 1, 1, 1], run='doe/hf_00', SEA=u['SEA'], ip=u['ip'], smax=u['smax'], plat_var=u['plat_var'],
                      eps_d=u['eps_d'], eroded=u['eroded'], tmin=cd.tmin([1, 1, 1, 1], 0.25), emulator=None)
for k, v in sel.items():
    pre = RUNS + 'confirm/' + k
    if not os.path.exists(pre + '_meta.json'):
        continue
    m = json.load(open(pre + '_meta.json'))
    if m['status'] != 'completed':
        continue
    r = record(pre)
    out[k] = dict(q=v['q'], run='confirm/' + k, SEA=r['SEA'], ip=r['ip'], smax=r['smax'], plat_var=r['plat_var'],
                  eps_d=r['eps_d'], eroded=r['eroded'], ne=r['ne'], tmin=cd.tmin(v['q'], 0.25), emulator=v['y_emulator'],
                  ke=r['ke_ie_max'], hg=r['hg_ie'])
json.dump(out, open(DATA + 'confirm_results.json', 'w'), indent=1)
for k, c in out.items():
    em = c['emulator']
    print('%-8s SEA %.2f%s ip %.2f%s var %.4f%s smax %.2f tmin %.2f eroded %d' % (k, c['SEA'], ' (%.2f)' % em[0] if em else '', c['ip'], ' (%.2f)' % -em[1] if em else '', c['plat_var'], ' (%.4f)' % 10**(-em[2]) if em else '', c['smax'], c['tmin'], c['eroded']))
