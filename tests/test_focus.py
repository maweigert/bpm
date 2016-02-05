"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt


from bpm import bpm_3d

from bpm import bpm_3d, focus_field_debye, psf_focus_u0

def test_focus(NA = .3, n0 = 1., Nx = 256, Nz = 256):
    """ propagates a focused wave freely to the center
    """


    dx, dz = .1, 0.1

    lam = .5

    units = (dx,dx,dz)

    _,u_debye,  _, _ = focus_field_debye((Nx, Nx, Nz), units, lam=lam, NAs=[0., NA])

    u0 = u_debye[0]

    u = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
                   n0 = n0,
                   u0 = u0,
                    dn = np.zeros((Nz,Nx,Nx)),
                    absorbing_width = 10)

    return u_debye, u

if __name__ == '__main__':

    # u1,u2 = test_plane(n_x_comp=1, n0 = 1.1)

    Nx  = 512
    Nz = 256

    NAs = [0.2,.4, .6, .8]
    n0s = [1.,1.,1.,1.]

    u_bpm, u_anal = [],[]
    for NA,n0 in zip(NAs, n0s):
        u1,u2 = test_focus(NA = NA, n0 = n0, Nx = Nx, Nz = Nz)
        u_bpm.append(u1)
        u_anal.append(u2)


    import pylab
    import seaborn
    col = seaborn.color_palette()

    n = len(u_bpm)


    pylab.figure(1)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,1,i+1)
        pylab.plot(np.real(u_anal[i][:,Nx/2,Nx/2]), "-",c = col[1],  label="analy")
        pylab.plot(np.real(u_bpm[i][:,Nx/2,Nx/2]), ".:", c = col[0], label="bpm")

        pylab.legend()
        pylab.title("NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

    pylab.figure(2)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,2,2*i+1)
        pylab.imshow(np.real(u_anal[i][:,Nx/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("anal,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

        pylab.subplot(n,2,2*i+2)
        pylab.imshow(np.real(u_bpm[i][:,Nx/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("bpm,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

    pylab.figure(3)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,2,2*i+1)
        pylab.imshow(np.abs(u_anal[i][Nz/2,...]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("anal,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

        pylab.subplot(n,2,2*i+2)
        pylab.imshow(np.abs(u_bpm[i][Nz/2,...]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("bpm,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))


    pylab.figure(4)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,2,2*i+1)
        pylab.imshow(np.abs(u_anal[i][:,Nx/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("anal,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

        pylab.subplot(n,2,2*i+2)
        pylab.imshow(np.abs(u_bpm[i][:,Nx/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("bpm,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))


    # pylab.subplot(2,1,2)
    # pylab.title("x_comp = 2")
    # pylab.plot(np.real(a1)[:,64,64], "-", c = col[1], label="2 bpm")
    # pylab.plot(np.real(a2)[:,64,64], ".",c = col[1],  label="2 analy")


    pylab.legend()
    pylab.show()
    pylab.draw()
    # test_plane(1)
    # test_plane(2)
    
