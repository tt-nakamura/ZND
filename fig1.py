import numpy as np
import matplotlib.pyplot as plt
from ZND import CJSlope, RayHugo

gam = 1.2
q = [10,20,30,40]
N = 100
EPS = 1e-8

plt.figure(figsize=(5, 3.75))

for q in q:
    u02 = CJSlope(q, gam)
    MCJ = np.sqrt(u02/gam) # CJ Mach number upstream
    M0 = np.geomspace(MCJ+EPS, 10, N)
    u02 = gam*M0**2
    v,p,u = RayHugo(q, u02, gam)
    M1 = u/np.sqrt(gam*p*v) # Mach number downstream
    plt.plot(M0, M1, label=r'$q=%d$'%q)
    plt.vlines(MCJ, 0, 1, linestyles='dotted')
    plt.ylim([0.3, 1.05])

plt.xlabel(r'$M_0$')
plt.ylabel(r'$M_1$')
plt.legend()
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
