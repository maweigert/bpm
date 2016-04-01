import numpy as np
from bpm import bpm_3d, psf_u0
import gputools
import pylab

def phi(dn,w):
    return w- np.arcsin(1./(1.+dn)*np.sin(w))


if __name__ == '__main__':

    Nx = 256
    Nz = 256
    dx = .1
    NA = .3
    lam = .5
    w = .8
    dn0 = 0.1

    x = dx*(np.arange(-Nx/2, Nx/2)+.5)
    y = dx*(np.arange(-Nx/2, Nx/2)+.5)
    z = dx*(np.arange(-Nz/2, Nz/2)+.5)

    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")

    Z,X = np.cos(w)*Z-np.sin(w)*X, np.cos(w)*X+np.sin(w)*Z

    dn = dn0*(Z>0)

    u0 = psf_u0((Nx,Nx), units = (dx,)*2,zfoc = dx*(Nz-1.)/2.,lam = lam, NA= NA)

    u = bpm_3d((Nx,Nx,Nz), units=(dx,)*3, lam=lam, dn =dn,
               u0=u0)

    u1 = np.abs(u[:Nz/2,...])[::-1,...]**2
    u2 = gputools.rotate(abs(u[Nz/2:,...])**2, center = (0,Nx/2,Nx/2),
                         axis = (0.,1.,0),
                         angle = phi(dn0,w),
                         mode = "linear")

    pylab.figure(1)
    pylab.clf()
    pylab.plot(u1[Nz/10,Nx/2,:])
    pylab.plot(u2[Nz/10,Nx/2,:])

    # return u, u0
    #
    #
    #
