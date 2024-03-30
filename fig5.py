import numpy as np
import matplotlib.pyplot as plt
from ZND import PathoDet

gam,Q1,Q2,E1,E2,k1,k2,N = 1.2, 60, -12, 60, 70, 5000, 5000, 100

lam1,lam2,t,x,v,p,u = PathoDet(gam,Q1,Q2,E1,E2,k1,k2,N=N)
T = v*p
c = np.sqrt(gam*T)
M = u/c
lam = (Q1*lam1 + Q2*lam2)/(Q1+Q2)
N2 = N//2

print(M[-1])

plt.figure(figsize=(5,10))

plt.subplot(5,1,1)
plt.plot(x, t)
plt.plot(x[N2], t[N2], 'k.')
plt.ylabel(r'$s$ = time')

plt.subplot(5,1,2)
plt.plot(x, u, label=r'$z$ = flow')
plt.plot(x, c, label=r'$\sqrt{\gamma vy}$ = sound')
plt.plot(x[N2], u[N2], 'k.')
plt.legend()
plt.ylabel(r'$z$, $\sqrt{\gamma vy}$ = velocities')

plt.subplot(5,1,3)
plt.plot(x, lam1, label=r'$\lambda_1$')
plt.plot(x, lam2, label=r'$\lambda_2$')
plt.plot(x, lam, label=r'$\lambda$')
plt.plot(x, M, label=r'Mach number')
plt.plot(x[N2], lam1[N2], 'k.')
plt.plot(x[N2], lam2[N2], 'k.')
plt.plot(x[N2], lam[N2], 'k.')
plt.plot(x[N2], M[N2], 'k.')
plt.legend()
plt.ylabel(r'$\lambda_1$, $\lambda_2$, $\lambda$, $M$')

plt.subplot(5,1,4)
plt.plot(x, T)
plt.plot(x[N2], T[N2], 'k.')
plt.ylabel(r'$vy$ = temperature')

plt.subplot(5,1,5)
plt.plot(x, p)
plt.plot(x[N2], p[N2], 'k.')
plt.ylabel(r'$y$ = pressure')

plt.xlabel(r'$x$ = distance')
plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
