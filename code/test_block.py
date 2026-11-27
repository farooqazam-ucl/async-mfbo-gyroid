import numpy as np, time, sys
from geometry import build_mesh
from fesolver import Simulation
# fully solid block: uniaxial stress test (frictionless platens, free lateral periods)
N, Nz, h = 4, 8, 1.0
mask = np.ones((N, N, Nz), dtype=bool)
mesh = build_mesh(mask, h); mesh['rho'] = 1.0; mesh['s'] = 0
mat = dict(sig0=30.0, Qv=20.0, bv=30.0)
H = Nz*h
c = np.sqrt(1650*(0.6)/(1.4*0.2)/9.3e-10)
T = 400 * 2*H/np.sqrt(1650/9.3e-10)
sim = Simulation(mesh, mat=mat, strain_max=0.10, T=T, mu_platen=0.0, mu_self=0.0, hist_dstrain=0.001)
t0=time.time(); r = sim.run(); print('wall %.1fs steps %d ne %d' % (time.time()-t0, r['steps'], r['ne']))
eps = r['strain']; sig = r['stress']
E = 1650.; nu = 0.4
# expected: uniaxial stress J2 with Voce; true vs nominal small difference
for e in [0.005, 0.01, 0.015, 0.02, 0.05, 0.08, 0.099]:
    i = np.argmin(abs(eps-e)); print('eps=%.3f sig=%.2f  Lx=%.4f  KE/IE=%.4f HG/IE=%.4f' % (eps[i], sig[i], r['Lx'][i], r['KE'][i]/max(r['IE'][i],1e-30), r['HG'][i]/max(r['IE'][i],1e-30)))
# analytical (nominal stress of uniaxial compression with true-stress Voce): approx
i = np.argmin(abs(eps-0.01)); print('apparent E (from 0.002-0.008):', np.polyfit(eps[(eps>0.002)&(eps<0.008)], sig[(eps>0.002)&(eps<0.008)],1)[0])
print('lateral strain at 0.008 (expect nu*eps=%.4f):' % (0.4*0.008), r['Lx'][np.argmin(abs(eps-0.008))]/4-1)
