"""


mweigert@mpi-cbg.de

"""


from bpm.bpmclass.bpm3d import Bpm3d
from numpy import *
import numpy as np

def get_prop(m, n0):
    H0 = np.sqrt(n0**2*m.k0**2-m._KX**2-m._KY**2)
    outsideInds = np.isnan(H0)

    H = np.exp(-1.j*m.units[-1]*H0)

    H[outsideInds] = 0.
    return H


def first_svd_comp(h):
    u,s,v = np.linalg.svd(h)
    u0 = np.sqrt(s[0])*u[:,0]
    v0 = np.sqrt(s[0])*v[0,:]
    return u0,v0, np.outer(u0,v0)


if __name__ == '__main__':

    Nx, Nz = 128,256
    dx = .1

    x = dx*(arange(Nx)-Nx/2)
    z = dx*(arange(Nz)-Nz/4)

    Z,Y,X = meshgrid(z,x,x,indexing="ij")
    R = sqrt(X**2+Y**2+Z**2)


    n = 1.+.1*(R<(dx*Nx/3))

    m = Bpm3d((Nx,Nx,Nz),(dx,)*3)


    n1 = 1.

    u1 = ones((Nz,Nx,Nx),complex64)


    for i in range(Nz-1):
        u1[i+1] = fft.fftn(u1[i])

        H = get_prop(m,n1)
        u1[i+1] *= H
        u1[i+1] = fft.ifftn(u1[i+1])
        u1[i+1] *= exp(-dx*1.j*m.k0*(n[i+1]-n1))

    n2 = mean(n,axis=(1,2))

    u2 = ones((Nz,Nx,Nx),complex64)


    for i in range(Nz-1):
        u2[i+1] = fft.fftn(u2[i])

        H = get_prop(m,n2[i])
        u2[i+1] *= H
        u2[i+1] = fft.ifftn(u2[i+1])
        u2[i+1] *= exp(-dx*1.j*m.k0*(n[i+1]-n2[i]))

