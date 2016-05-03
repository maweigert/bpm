

import numpy as np
import numpy.testing as npt


from bpm import bpm_3d

from bpm import bpm_3d


if __name__ == '__main__':

    Nx, Ny, Nz = 256,256,512

    dx, dz = .05, 0.05

    lam = .5

    units = (dx,dx,dz)

    x = dx*np.arange(-Nx/2,Nx/2)
    y = dx*np.arange(-Ny/2,Ny/2)
    z = dz*np.arange(-Nz/2,Nz/2)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")
    R = np.sqrt(X**2+Y**2+Z**2)
    dn = .05*(R<1.)

    u1 = bpm_3d((Nx,Ny,Nz),units= units, lam = lam,
                   dn = dn,
                   n_volumes = 2,
                    return_field=True)

    u1 = abs(u1)**2
    u2 = bpm_3d((Nx,Ny,Nz),units= units, lam = lam,
                   dn = dn,
                   n_volumes = 2,
                    return_field=False)

    is_close = np.allclose(u1,u2)

    print is_close
    assert is_close