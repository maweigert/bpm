


from bpm import bpm_3d


import sys
import pymiecoated as pym


from numpy import *

# the mie code function
def qsca(r1, dn1 = .1, r2 = None, dn2 = None, lam = .5, n0 = 1.,
         return_g = False):

    if r2 is None:
        r2 = r1
    if dn2 is None:
        dn2 = dn1

    # size parameters
    x = 2*pi*r1/lam*n0
    y = 2*pi*r2/lam*n0

    # the relative r.i.
    m = 1.+1.*dn1/n0
    m2 = 1.+1.*dn2/n0

    m = pym.Mie(x = x,m=m, y=y,m2 = m2)
    if return_g:
        return m.qsca(), m.asy()
    else:
        return m.qsca()

def gfac(r1, dn1 = .1, r2 = None, dn2 = None, lam = .5, n0 = 1.):

    if r2 is None:
        r2 = r1
    if dn2 is None:
        dn2 = dn1

    # size parameters
    x = 2*pi*r1/lam*n0
    y = 2*pi*r2/lam*n0

    # the relative r.i.
    m = 1.+1.*dn1/n0
    m2 = 1.+1.*dn2/n0

    m = pym.Mie(x = x,m=m, y=y,m2 = m2)
    return m.asy()



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


def bpm_sca(r1,dn1,r2 = None,dn2 = None, lam= .5, n0 = 1., return_g = False):

    if dn2 is None:
        dn2 = dn1
    if r2 is None:
        r2 = r1
        r1 = .5*r2

    Nz, Nx = 256,256
    dx = .1
    dn0 = zeros((Nz,Nx,Nx),float32)

    add_coated_sphere(dn0,[Nz/2,Nx/2,Nx/2],r1=int(r1/dx),r2 = int(r2/dx),dn1 = dn1,dn2 = dn2)

    res = bpm_3d((Nx,Nx,Nz),units= (dx,)*3,
                            lam = lam,
                            n0 = n0,
                            dn = dn0,
                            return_scattering = True,
                        return_g = return_g
                        )
    if return_g:
        p, g = res[-2:]
        return p[-1]/r**2/pi, g[-1]
    else:
        p = res[-1]
        return p[-1]/r**2/pi

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
    return p, dn0, g

if __name__ == '__main__':

    lam = .5
    n0 = 1.3
    
    dns = linspace(.01,.2,20)

    dn = 0.02

    rs = linspace(1.,10.,100)
    #rs = linspace(1.,20.,60)

    p, dn0, g = bpm_p(2.5,n0*0.04, r2 = 3., dn2 = n0*0.02,lam = lam,n0= n0)
    #
    p_mie, g_mie = array([qsca(r1=.8*r,dn1=dn,r2=r,dn2 = 1.2*dn,
                               lam = lam,n0 = n0, return_g  = True) for r in rs]).T


    p_bpm, g_bpm = array([bpm_sca(r1=.8*r,dn1=dn,r2=r,dn2 = 1.2*dn,
                                  lam = lam,n0= n0, return_g = True) for r in rs]).T



    # p_mie2, g_mie2 = array([qsca(r1=r,dn1=dn,
    #                            lam = lam,n0 = n0, return_g  = True) for r in rs]).T
    # #
    # p_bpm2, g_bpm2 = array([bpm_sca(r,dn, lam = lam,n0= n0, return_g = True) for r in rs]).T

