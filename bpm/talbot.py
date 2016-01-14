
from bpm.bpm.bpm_3d import bpm_3d

from numpy import *
from scipy.special import jn

def overlap(y1,y2):
    m1 = sqrt(mean(y1**2))
    m2 = sqrt(mean(y2**2))
    return mean(y1*y2)/m1/m2
    
    
if __name__ == '__main__':

    lam = .5
    Lx  = 20.
    a = Lx/4.

    Lz = 2*a**2/lam
    Lz = lam/(1.-sqrt(1.-lam**2/a**2))

    Lz = 1.5*Lz
    Nx = 512
    
    Nz = 100

    dx = 1.*Lx/Nx
    dz = 1.*Lz/(Nz-1)

    
    lam = .5

    x = dx*arange(Nx)
    y = dx*arange(Nx)
    z = dz*arange(Nz)

    X0,Y0 = meshgrid(x,y,indexing="ij")



    _xgrid = (X0%a)-.5*a
    _ygrid = (Y0%a)-.5*a

    R = sqrt(_xgrid**2+_ygrid**2)

    
    u0 = (jn(1,4*R)+2.e-10)/4./(1.e-10+R)+0.j
    u0 = u0.astype(complex64)

    # u0 = exp(-20*(_xgrid**2+_ygrid**2))

    # u0 *= exp(-1.5*(R0/dx/Nx*3.)**2)

    u, dn = bpm_3d((Nx,Nx,Nz), (dx,dx,dz), lam = lam ,u0 = u0,
                   use_fresnel_approx = False)

    overlaps = array([overlap(abs(u[0,...])**2,abs(u[i,...])**2) for i in range(Nz)])
