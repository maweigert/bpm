"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt



from bpm import bpm_3d, psf_debye, psf_focus_u0


if __name__ == '__main__':

    # u1,u2 = test_plane(n_x_comp=1, n0 = 1.1)

    Nx  = 256
    Nz = 512
    NA = .7


    _,u1, _, _ = psf_debye((Nx,Nx,2*Nz),(.1,)*3,lam = .5,NAs = [0.,NA])

    u1 = u1[Nz:]

    u2 = bpm_3d((Nx,Nx,Nz),(.1,)*3,u0 = u1[0])

    u3 = bpm_3d((Nx,Nx,Nz),(.1,)*3,u0 = u1[0],dn = np.zeros((Nz,Nx,Nx)),
                absorbing_width=10)

    import pylab
    pylab.figure()
    pylab.clf()

    for i,u in enumerate([u1,u2,u3]):
        pylab.subplot(1,3,i+1)
        pylab.imshow(abs(u[...,Nx/2])**.6,cmap = "hot")
        pylab.axis("off")

    pylab.show()