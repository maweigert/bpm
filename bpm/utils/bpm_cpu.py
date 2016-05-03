"""


mweigert@mpi-cbg.de

"""

import bpm
from bpm.bpmclass.bpm3d import Bpm3d
from numpy import *
import numpy as np
import pylab
import gputools
from time import time

if __name__ == '__main__':

    Nx, Nz = 256,512
    dx = .1


    x = dx*np.linspace(-Nx,Nx,Nx+1)[:Nx]
    z = dx*np.linspace(-Nz,Nz,Nz+1)[:Nz]
    Z,Y,X = np.meshgrid(z,x,x,indexing = "ij")

    R = sqrt(X**2+Y**2+(Z-9)**2)

    R2 = sqrt(X**2+Y**2+(Z+9)**2)

    n = 1.+.1*(R<(dx*Nx/4))
    n += .1*(R2<(dx*Nx/4))


    # n = 1.+.1*(maximum(abs(X),abs(Y))<(dx*Nx/3))*(abs(Z)<dx*Nz/4)


    #dn[Nz/2:,...] = 0.0

    m = Bpm3d((Nx,Nx,Nz),(dx,)*3)


    def prop(n0):
        H0 = np.sqrt(n0**2*m.k0**2-m._KX**2-m._KY**2)
        outsideInds = np.isnan(H0)

        H = np.exp(-1.j*dx*H0)

        H[outsideInds] = 0.
        return H


    u0 = bpm.psf_u0((Nx,Nx),(dx,dx),dx*Nz/2.,NA=.3)
    u0 *= 1./np.sqrt(np.mean(abs(u0)**2))
    #u0 = ones((Nx,Nx),complex64)

    u1 = ones((Nz,Nx,Nx),complex64)

    u1[0] = u0
    H1 = prop(1.)
    t = time()
    n0 = 1.
    for i in range(Nz-1):
        _u = u1[i]
        _u = fft.fftn(_u)
        _u *= H1
        _u = fft.ifftn(_u)
        _u *= exp(-dx*1.j*m.k0*(n[i+1]-n0))
        u1[i+1] = _u

    print "time: %.1f ms"%(1000*(time()-t))


    u2 = ones((Nz,Nx,Nx),complex64)
    u2[0] = u0
    n_mean = np.mean(n,axis=(1,2))
    t = time()
    for i in range(Nz-1):
        n0 = mean(n[i+1])
        H2 = prop(n0)
        _u = u2[i]
        _u = fft.fftn(_u)
        _u *= H2
        _u = fft.ifftn(_u)
        _u *= exp(-dx*1.j*m.k0*(n[i+1]-n0))
        u2[i+1] = _u

    print "time: %.1f ms"%(1000*(time()-t))

    u3 = ones((Nz,Nx,Nx),complex64)
    u3[0] = u0
    t = time()
    for i in range(Nz-1):
        _u = u3[i]
        #print mean(abs(_u**2))
        n0 = mean(abs(_u)**2*n[i+1])/mean(abs(_u**2))
        H2 = prop(n0)

        _u = fft.fftn(_u)
        _u *= H2
        _u = fft.ifftn(_u)
        _u *= exp(-dx*1.j*m.k0*(n[i+1]-n0))
        u3[i+1] = _u

    print "time: %.1f ms"%(1000*(time()-t))

    #
    # pylab.clf()
    # pylab.plot(mean(abs(u)**2,(1,2)))
    # pylab.show()