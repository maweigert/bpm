


from bpm import bpm_3d


import sys
import pymiecoated as pym


from numpy import *

def add_coated_sphere(dn,
                      c = (100,100,100),
                      r1 = 8,
                      r2 = 11,
                      dn1 = 0.04,
                      dn2 = 0.02):
    """
    r1 is the inner radius, r2 the outer
    """

    assert(r2>=r1)

    r1 = int(r1)
    r2 = int(r2)
    x = arange(-r2,r2+1)
    Z,Y,X = meshgrid(x,x,x,indexing="ij")
    R = sqrt(X**2+Y**2+Z**2)
    m_outer = R<=r2
    m_inner = R<=r1
    ss = tuple([slice(_c-r2,_c+r2+1) for _c in c])
    dn[ss][m_outer] = dn2
    dn[ss][m_inner] = dn1


def bpm_p(r1,dn1, r2 = None, dn2 = None, lam= .5, n0 = 1.):
    if dn2 is None:
        dn2 = dn1
    if r2 is None:
        r2 = r1
        r1 = .5*r2

    Nz, Nx = 256,256
    dx = .1
    dn0 = zeros((Nz,Nx,Nx),float32)

    add_coated_sphere(dn0,[Nz/2,Nx/2,Nx/2],r1=int(r1/dx),r2 = int(r2/dx),dn1 = dn1,dn2 = dn2)

    u, _, p, g = bpm_3d((Nx,Nx,Nz),units= (dx,)*3,
                            lam = lam,
                            n0 = n0,
                            dn = dn0,
                            return_scattering = True,
                     return_g = True
                            )
    return u, p, dn0, g

if __name__ == '__main__':

    lam = .5
    n0 = 1.

    dn = 0.02
    


    u, p, dn0, g = bpm_p(2.5,n0*dn, r2 = 3., dn2 = n0*dn,lam = lam,n0= n0)
