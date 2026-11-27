"""Run one FE crush analysis of a graded sheet-gyroid column."""
import argparse, json, os, sys, time
import numpy as np
from geometry import make_design_mesh, VoxelGrid
from fesolver import Simulation

MAT = dict(E=1650.0, nu=0.40, rho=9.30e-10, sig0=30.0, Qv=20.0, bv=30.0, Hl=0.0)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--q', type=float, nargs=4, default=[1, 1, 1, 1])
    p.add_argument('--N', type=int, default=24)
    p.add_argument('--a', type=float, default=25.0)
    p.add_argument('--nz', type=int, default=4)
    p.add_argument('--rho', type=float, default=0.25)
    p.add_argument('--Tf', type=float, default=20.0)
    p.add_argument('--emax', type=float, default=0.6)
    p.add_argument('--mu_p', type=float, default=0.2)
    p.add_argument('--mu_s', type=float, default=0.2)
    p.add_argument('--sig0', type=float, default=None)
    p.add_argument('--Qv', type=float, default=None)
    p.add_argument('--hg', type=float, default=1.0/48.0)
    p.add_argument('--ramp', type=float, default=0.04)
    p.add_argument('--hgfloor', type=float, default=0.02)
    p.add_argument('--epsfail', type=float, default=0.0)
    p.add_argument('--limit', type=float, default=None)
    p.add_argument('--stopfile', type=str, default=None)
    p.add_argument('--snaps', type=float, nargs='*', default=[])
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args(argv)
    mat = dict(MAT)
    if a.sig0 is not None: mat['sig0'] = a.sig0
    if a.Qv is not None: mat['Qv'] = a.Qv
    mat['eps_fail'] = a.epsfail
    mesh, mask = make_design_mesh(a.q, a.rho, a.a, a.N, a.nz)
    sim = Simulation(mesh, mat=mat, strain_max=a.emax, T_factor=a.Tf, mu_platen=a.mu_p,
                     mu_self=a.mu_s, hg_coef=a.hg, sigma_limit=a.limit, stop_file=a.stopfile,
                     snap_strains=a.snaps, verbose=a.verbose, ramp_frac=a.ramp, hg_floor=a.hgfloor)
    res = sim.run(out_prefix=a.out)
    meta = json.load(open(a.out + '_meta.json'))
    meta.update(q=list(a.q), N=a.N, a=a.a, nz=a.nz, rho_target=a.rho, Tf=a.Tf, mat=mat,
                mu_p=a.mu_p, mu_s=a.mu_s, hg=a.hg, limit=a.limit, ramp=a.ramp, hgfloor=a.hgfloor)
    json.dump(meta, open(a.out + '_meta.json', 'w'), indent=1)
    print('done %s status=%s wall=%.1fs steps=%d ne=%d' % (a.out, res['status'], res['wall'], res['steps'], res['ne']), flush=True)

if __name__ == '__main__':
    main()
