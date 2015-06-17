
import numpy as np

from bpm.utils import StopWatch, absPath
from bpm.bpm_3d import bpm_3d


import sys
sys.path.append("/Users/mweigert/python/mie_scat")

from bhmie_code import bhmie

def getMieEff(r, lam = .5,n= 1.1):
    S1,S2,Qext, Qsca,Qback,gsca = bhmie(2*pi*r/lam,n,128)
    return Qsca


if __name__ == '__main__':
    # test_speed()

    
    Nx, Nz = 256,256
    dx, dz = .1, .1

    lam = .5

    units = (dx,dx,dz)

    
    x = dx*np.arange(-Nx/2,Nx/2)
    x = dx*np.arange(-Nx/2,Nx/2)
    z = dz*np.arange(-Nz/2,Nz/2)
    Z,Y,X = np.meshgrid(z,x,x,indexing="ij")
    R = np.sqrt(X**2+Y**2+Z**2)

    rad = 3.
    dn = 0.05*(R<rad)

    u, dn, p = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
                      dn = dn,
                      return_scattering = True )

    ps = []
    rads = np.linspace(1.,6.,10)
    for rad in rads:
        print rad
        dn = 0.05*(R<rad)
    
        u, dn, p = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
                      dn = dn,
                      return_scattering = True )

        ps.append(p[-1])
    

    ps = np.array(ps)
