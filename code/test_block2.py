import numpy as np, time
from geometry import build_mesh
from fesolver import Simulation
N, Nz, h = 4, 8, 1.0
mask = np.ones((N, N, Nz), dtype=bool)
for baro, kpen in [(True,0.15),(True,1.0),(False,0.15)]:
    mesh = build_mesh(mask, h); mesh['rho'] = 1.0
    T = 400 * 2*Nz*h/np.sqrt(1650/9.3e-10)
    sim = Simulation(mesh, mat=dict(sig0=1e6), strain_max=0.01, T=T, mu_platen=0.0, mu_self=0.0, hist_dstrain=0.0005, barostat=baro, kpen=kpen)
    r = sim.run()
    eps, sig = r['strain'], r['stress']; sel=(eps>0.003)&(eps<0.009)
    print('baro',baro,'kpen',kpen,'E_app=%.0f' % np.polyfit(eps[sel], sig[sel],1)[0], 'sigbot/sigtop=%.3f'%(r['stress_bot'][-1]/r['stress'][-1]), 'lat=%.5f'%(r['Lx'][-1]/4-1), 'mean sxx=%.3f szz=%.3f'%(sim.sig[:,0].mean(), sim.sig[:,2].mean()), 'ezz_mat=%.5f'%((sim.x[:,2].max()-sim.x[:,2].min())/8-1))
