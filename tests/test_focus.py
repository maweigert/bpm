"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt


from bpm import bpm_3d, psf, psf_u0

def test_focus(size, units, NA = .3, n0 = 1.):
    """ propagates a focused wave freely to the center
    """

    Nx, Ny, Nz = size

    dx, dy , dz = .1, .1, .1

    lam = .5

    _,u_debye,  _, _ = psf(size, units, n0= n0, lam=lam, NA=NA, return_field = True)

    u0 = u_debye[0]
    u0 = psf_u0(size[:2],units[:2],zfoc = .5*units[-1]*(size[-1]-1), lam = lam, NA = NA)

    u = bpm_3d(size,units= units, lam = lam,
                   n0 = n0,
                   u0 = u0,
                   absorbing_width = 0)

    return u, u_debye

if __name__ == '__main__':


    Nx  = 256
    Ny = 512
    Nz = 256

    dx, dy, dz = .1, .1, .1


    NAs = [0.2,.4, .6]
    n0s = [1.,1.2,1.4]

    u_bpm, u_anal = [],[]
    for NA,n0 in zip(NAs, n0s):
        u1,u2 = test_focus((Nx,Ny,Nz),NA = NA, n0 = n0)
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
        pylab.plot(np.real(u_anal[i][:,Ny/2,Nx/2]), "-",c = col[1],  label="analy")
        pylab.plot(np.real(u_bpm[i][:,Ny/2,Nx/2]), ".:", c = col[0], label="bpm")

        pylab.legend()
        pylab.title("NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

    pylab.figure(2)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,2,2*i+1)
        pylab.imshow(np.real(u_anal[i][:,Ny/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("anal,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

        pylab.subplot(n,2,2*i+2)
        pylab.imshow(np.real(u_bpm[i][:,Ny/2,:]), cmap = "hot")
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
        pylab.imshow(np.abs(u_anal[i][:,Ny/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("anal,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))

        pylab.subplot(n,2,2*i+2)
        pylab.imshow(np.abs(u_bpm[i][:,Ny/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.axis("off")
        pylab.title("bpm,  NA = %s, n0 = %.2f"%(NAs[i],n0s[i]))



    pylab.legend()
    pylab.show()
    pylab.draw()

