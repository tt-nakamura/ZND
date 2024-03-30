import numpy as np
import matplotlib.pyplot as plt
from ZND import ZND

gam,Q,E,k = 1.2, 50, 50, 1000

lam,t,x,v,p,u = ZND(gam,Q,E,k)
T = v*p
c = np.sqrt(gam*T)

plt.figure(figsize=(5,10))

plt.subplot(5,1,1)
plt.plot(x, t)
plt.ylabel(r'$s$ = time')

plt.subplot(5,1,2)
plt.plot(x, u, label=r'$z$ = flow')
plt.plot(x, c, label=r'$\sqrt{\gamma vy}$ = sound')
plt.legend()
plt.ylabel(r'$z$, $\sqrt{\gamma vy}$ = velocities')

plt.subplot(5,1,3)
plt.plot(x, lam, label=r'$\lambda$')
plt.plot(x, u/c, label=r'Mach number')
plt.legend()
plt.ylabel(r'$\lambda$, $M$')

plt.subplot(5,1,4)
plt.plot(x, T)
plt.ylabel(r'$vy$ = temperature')

plt.subplot(5,1,5)
plt.plot(x, p)
plt.ylabel(r'$y$ = pressure')

plt.xlabel(r'$x$ = distance')
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
