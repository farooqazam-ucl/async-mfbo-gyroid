"""Write data/slab_profiles.json from the snapshot files of the raw-data archive.

The file stores, for every analysis with saved snapshots and every saved
strain, the mean equivalent plastic strain in ten slabs of the undeformed
column height.  make_macros.py reads these profiles when the snapshot files
(runs/*/*_snaps.npz, about 150 MB) are not present, so that every number in
the paper can be regenerated from this repository alone.

Usage (after unzipping the raw-data archive into the repository folder):
    python code/make_slab_cache.py
"""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import glob
import json
import numpy as np

RUNS = _ROOT + 'runs/'
DATA = _ROOT + 'data/'


def profiles(fn, nb=10):
    S = np.load(fn)
    eijk = S['eijk']
    nz = eijk[:, 2].max() + 1
    idx = np.minimum(((eijk[:, 2] + 0.5) / nz * nb).astype(int), nb - 1)
    out = {}
    for k in S.files:
        if k.startswith('ep_'):
            ep = S[k]
            out[k[3:]] = [float(ep[idx == i].mean()) for i in range(nb)]
    return out


def main(nb=10):
    files = sorted(glob.glob(RUNS + '*/*_snaps.npz'))
    if not files:
        raise SystemExit('No snapshot files found under runs/.  Unzip the raw-data archive first.')
    cache = {}
    for fn in files:
        run = fn[len(RUNS):-len('_snaps.npz')]
        for key, prof in profiles(fn, nb).items():
            cache['%s|%s|%d' % (run, key, nb)] = prof
    with open(DATA + 'slab_profiles.json', 'w') as fh:
        json.dump(cache, fh, indent=0)
    print('%d profiles from %d snapshot files written to data/slab_profiles.json' % (len(cache), len(files)))


if __name__ == '__main__':
    main()
