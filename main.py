import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

def plotit(l,xmax = 20):
    x = np.linspace(0,xmax,1000)
    y = sp.spherical_jn(l,x,derivative = False)
    yp = sp.spherical_jn(l,x,derivative = True)
    #plt.plot(x,y)
    plt.plot(x,yp)

def getzeros(l,nz,eps = 1e-16):
    xguess = xguess_n(l,nz)
    return xgood(l,xguess,eps)

def fp(x,l):
    if l==0:
        return -sp.spherical_jn(1,x,derivative=True)
    else:
        return sp.spherical_jn(l-1,x,derivative=True)-(l+1)*sp.spherical_jn(l,x,derivative=True)/x+(l+1)*sp.spherical_jn(l,x,derivative=False)/x**2

def xguess_n(l,nt,step = 10):
    #finds a rough guess of where the first nt zeros of the lth spherical bessel function derivative are by finding points where the sign flips
    x0 = 0
    n = 0
    ps = 0
    i = 0
    while n<nt:
        xmin = x0 + i*step
        xmax = x0 + (i+1)*step
        p = xguess_int(l,xmin,xmax)
        if np.shape(ps) == ():
            ps = p
        else:
            ps = np.concatenate((ps,p))
        n = ps.size
        i = i+1

    return ps

def xguess_int(l,xmin,xmax):
    #finds a rough guess of where zeros of the lth spherical bessel function derivative are in the interval [xmin,xmax] by finding points where the sign flips
    x = np.linspace(xmin,xmax,1000)
    y = sp.spherical_jn(l,x,derivative=True)
    z = (np.roll(y,1)*y)<0
    if l==0 and xmin==0:
        z[0] = True
    else:
        z[0] = False
    return x[z]

def xgood(l,xguess,epsilon = 0):
    #uses rootfinding to get higher precision estimates of the zeros
    good = np.zeros(xguess.shape)
    f = lambda x:sp.spherical_jn(l,x,derivative=True)
    f2 = lambda x:fp(x,l)
    for i in range(xguess.size):
        if epsilon == 0:
            g = opt.root_scalar(f,method = 'newton',fprime = f2,x0 = xguess[i])
        else:
            g = opt.root_scalar(f,method = 'newton',fprime = f2,x0 = xguess[i],xtol=epsilon)

        good[i] = g.root
        
    return good
