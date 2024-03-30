import numpy as np
import matplotlib.pyplot as plt
from ZND import ZND

gam,Q = 1.2, 50
E = [30,40,50,60,70]

plt.figure(figsize=(5,3.75))

for E in E:
    lam,t,x,v,p,u = ZND(gam,Q,E)
    xf = x[np.argmin(np.abs(lam - 0.5))]
    plt.plot(x/xf, v*p, label=r'$\epsilon$=%d'%E)

plt.ylabel(r'$vy$ = temperature')
plt.xlabel(r'$x/x_{1/2}$ = distance')
plt.xlim([0,2])
plt.legend()
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
