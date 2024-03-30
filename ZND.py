# referece:
# J.H.S. Lee, "The Detonation Phenomenon" chapters 2 and 4

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq

def CJSlope(Q, gam=1.2, gam0=None):
    """ Chapman-Jouguet detonation wave velocity
    = slope of Rayleigh line that is tangent to Hugoniot curve.
    Q: heat released from combustion / (p0/rho0)
    gam,gam0: specific heat ratio c_p/c_v
        after and before combustion
    return u0^2/(p0/rho0) = gam*M0^2 = slope of the tangent
    If gam0 is None, gam0 is set to gam.
    reference: J.H.S.Lee, section 2.4
    """
    if gam0 is None: gam0 = gam
    B = (gam**2 - 1)*(1/(gam0-1) + Q) - 1
    D = (B - gam)*(B + gam)
    return B + np.sqrt(D)

def RayHugo(Q, u02, gam=1.2, gam0=None, sign=-1):
    """ intersection of Rayleigh line and Hugoniot curve
    Q = heat released from combustion / (p0/rho0)
    u02: slope of Rayleigh line u0^2/(p0/rho0)
    gam,gam0: specific heat ratio c_p/c_v
        after and before combustion
    sign: sign of square root, sign<0 or sign>0 for
        ordinary or pathological detonation, respectively.
    return v,p,u where
    v: specific volume / (1/rho0),
    p: pressure / p0,
    u: flow velocity / sqrt(p0/rho0).
    If gam0 is None, gam0 is set to gam.
    reference: J.H.S.Lee, section 2.3
    """
    Gam = 2*gam/(gam-1)
    Gam0 = Gam if gam0 is None else 2*gam0/(gam0-1)
    b = (Gam*(1+u02)/2)**2
    c = (Gam-1)*u02*(Gam0 + u02 + 2*Q)
    d = np.asarray(b-c);
    d[d<0 & np.isclose(b,c)] = 0 # to avoid rounding error
    d = np.sqrt(d)
    if sign>0: d=-d
    v = (gam*(1+u02) - (gam-1)*d)/(gam+1)/u02
    p = 1 + u02*(1-v)
    u = np.sqrt(u02)*v
    return v,p,u

def ZND(gam, Q, E, k=1, gam0=None, u02=None,
        lam1=0.9999, N=100):
    """ Zeldovich-Neumann-Doering detonation
    gam: spcifice heat ratio c_p/c_v after combustion
    Q: heat released from combustion / (p0/rho0)
    E: energy barrier of chemical reaction / (p0/rho0)
    k: (time scale of chemical reaction)^{-1}
    u02 = slope of Rayleigh line u0^2/(p0/rho0)
    gam0: spcifice heat ratio c_p/c_v before combustion
    lam1: end of integral interval [0,lam1] for lambda
    N: number of plotting points
    return lam,t,x,v,p,u, each shape(N,), where
    lam: progress parameter of chmical reaction
    t,x: time and distance from shock plane
    v: specfic volume, p: pressure, u: flow velocity.
    Variables are nondimensionalized by setting rho0=p0=1.
    Assume Arrhenius law d(lam)/dt = k(1-lam)e^{-E/RT},
       and ideal equation of state RT = p/rho.
    If gam0 is None, gam0 is set to gam.
    If u02 is None, u02 is set by CJSlope().
    u02 must be >= CJSlope(), lam1 must be < 1.
    reference: J.H.S.Lee, section 4.2
    """
    uc2 = CJSlope(Q, gam, gam0)
    if u02 is None: u02 = uc2
    elif u02 < uc2:
        raise RuntimeError('too small u02 in ZND')

    def difeq(y,lam):
        v,p,u = RayHugo(Q*lam, u02, gam, gam0)
        T = p*v
        dt = np.exp(E/T)/(1 - lam)/k
        dx = u*dt
        return dt,dx

    N2 = N//2
    lam = np.r_[np.linspace(0, 0.5, N2, endpoint=0),
                1 - np.geomspace(0.5, 1-lam1, N-N2)]
    t,x = odeint(difeq, [0,0], lam).T
    v,p,u = RayHugo(Q*lam, u02, gam, gam0)
    return lam,t,x,v,p,u

def PathoDet(gam, Q1, Q2, E1, E2, k1=1, k2=1,
             gam0=None, lam1=0.99999, N=100):
    """ Pathological Detonation
    gam: spcifice heat ratio c_p/c_v after combustion
    Q1: heat released from combustion1 (Q1>0)
    Q2: -(heat absorbed from combustion2) (Q2<0)
    E1,E2: energy barriers of each chemical reactions (E1<E2)
    k1,k2: (time scales of each chemical reactions)^{-1}
    gam0: spcifice heat ratio c_p/c_v before combustion
    lam1: end of integral interval [0,lam1] for lambda
    N: number of plotting points
    return lam,mu,t,x,v,p,u, each shape(N,), where
    lam,mu: progress parameters of each chmical reactions
    t,x: time and distance from shock plane
    v: specfic volume, p: pressure, u: flow velocity.
    Variables are nondimensionalized by setting rho0=p0=1.
    Assume Arrhenius law
       d(lam)/dt = k1(1-lam)e^{-E1/RT},
       d(mu)/dt = k2(lam-mu)e^{-E2/RT},
       and ideal equation of state RT = p/rho.
    If gam0 is None, gam0 is set to gam.
    Sonic point corresponds to (N//2)th element of output arrays.
    reference: J.H.S.Lee, section 4.3
    """
    if Q1<=0 or Q2>=0 or E1>=E2:
        raise RuntimeError('wrong Q or E in PathoDet')

    def difeq(y, u, u02):
        lam,mu = y
        v = u/np.sqrt(u02)
        p = 1 + u02*(1-v)
        T = p*v
        dlm = k1*(1-lam)*np.exp(-E1/T)
        dmu = k2*(lam-mu)*np.exp(-E2/T)
        dQ = Q1*dlm + Q2*dmu
        if dQ<=0: raise Exception
        dt = (gam*T - u**2)/(gam-1)/u/dQ
        return dlm*dt, dmu*dt

    def func(u02):
        u0 = np.sqrt(u02)
        v = gam/(gam+1)*(1+u02)/u02
        u2 = v*u0 # at sonic point
        u1 = (2*gam + (gam-1)*u02)/(gam+1)/u0 # behind shock

        try: y = odeint(difeq, [0,0], [u1,u2], args=(u02,))
        except: return -1

        global lam
        lam,mu = y[-1] # at sonic point
        p = 1 + u02*(1-v) # pressure
        T = p*v # temperature
        dlm = k1*(1-lam)*np.exp(-E1/T)
        dmu = k2*(lam-mu)*np.exp(-E2/T)
        return Q1*dlm + Q2*dmu

    a = CJSlope(Q1+Q2, gam, gam0)
    b = CJSlope(Q1, gam, gam0)
    u02 = brentq(func, a, b)

    def difeq(y, lam, sign):
        t,x,mu = y
        Q = Q1*lam + Q2*mu
        v,p,u = RayHugo(Q, u02, gam, gam0, sign)
        T = p*v
        dt = np.exp(E1/T)/k1/(1-lam)
        dmu = k2*(lam-mu)*np.exp(-E2/T)*dt
        dx = u*dt
        return dt,dx,dmu

    N2 = N//2
    la = np.linspace(0, lam, N2+1) # global lam
    lb = 1 - np.geomspace(1-lam, 1-lam1, N-N2)
    ya = odeint(difeq, [0,0,0], la, args=(-1,))
    yb = odeint(difeq, ya[-1], lb, args=(1,))
    lb,yb = lb[1:],yb[1:]
    ta,xa,ma = ya.T
    tb,xb,mb = yb.T
    Qa = Q1*la + Q2*ma
    Qb = Q1*lb + Q2*mb
    va,pa,ua = RayHugo(Qa, u02, gam, gam0, -1)
    vb,pb,ub = RayHugo(Qb, u02, gam, gam0, 1)
    return np.c_[np.r_[la,lb],
                 np.r_[ma,mb],
                 np.r_[ta,tb],
                 np.r_[xa,xb],
                 np.r_[va,vb],
                 np.r_[pa,pb],
                 np.r_[ua,ub]].T
