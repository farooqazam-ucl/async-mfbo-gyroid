"""Select representative Pareto designs from all campaigns for FE confirmation."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import sys, json, glob
import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
DATA = _ROOT + 'data/'

obs = []
for f in sorted(glob.glob(DATA + 'campaigns/*.json')):
    r = json.load(open(f))
    if r['arm'] not in ('A', 'A0', 'B', 'B0'):  # campaigns of Modes A and B (complete when the re-analyses were set up)
        continue
    for o in r['obs']:
        if o['lam'] == 1 and o['status'] == 'completed' and o['feasible']:
            obs.append(o)
Y = np.array([o['y'] for o in obs])          # max form [SEA, -ip, -logvar]
nd = is_non_dominated(torch.as_tensor(Y)).numpy()
P = [obs[i] for i in np.nonzero(nd)[0]]; YP = Y[nd]
lo, hi = YP.min(0), YP.max(0)
Z = (YP - lo) / np.maximum(hi - lo, 1e-9)
pick = {}
pick['maxSEA'] = int(np.argmax(YP[:, 0]))
pick['minIP'] = int(np.argmax(YP[:, 1]))
pick['minVAR'] = int(np.argmax(YP[:, 2]))
pick['knee'] = int(np.argmin(np.linalg.norm(1 - Z, axis=1)))
sel = {}
for k, i in pick.items():
    q = np.array(P[i]['q']); q = q / q.mean()
    sel[k] = dict(q=q.tolist(), y_emulator=list(map(float, YP[i])))
    print(k, np.round(q, 3), np.round(YP[i], 3))
json.dump(dict(selected=sel, n_pareto=int(nd.sum()), n_feasible_hf=len(obs)), open(DATA + 'confirm_designs.json', 'w'), indent=1)
