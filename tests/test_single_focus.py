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


    Nx = 128
    Ny = 256
    Nz = 256

    dx ,dy, dz = .2, .1, .1
    NA = .6
    n0 = 1.


    u_bpm,u_anal = test_focus((Nx,Ny,Nz),(dx,dy,dz),NA = NA, n0 = n0)


    print "L2 difference = %.10f"%np.mean(np.abs(u_bpm-u_anal)**2)



    import pylab
    import seaborn
    col = seaborn.color_palette()

    pylab.figure(1)
    pylab.clf()
    pylab.subplot(1,3,1)
    pylab.imshow(np.abs(u_anal[:,Ny/2,:]), cmap = "hot")
    pylab.grid("off")
    pylab.axis("off")
    pylab.title("anal,  NA = %s, n0 = %.2f"%(NA,n0))

    pylab.subplot(1,3,2)
    pylab.imshow(np.abs(u_bpm[:,Ny/2,:]), cmap = "hot")
    pylab.grid("off")
    pylab.axis("off")
    pylab.title("bpm,  NA = %s, n0 = %.2f"%(NA,n0))
    pylab.subplot(1,3,3)
    pylab.imshow(np.abs(u_bpm[:,Ny/2,:]-u_anal[:,Ny/2,:]), cmap = "gray")
    pylab.grid("off")
    pylab.axis("off")
    pylab.title("difference")


    pylab.figure(2)
    pylab.clf()

    pylab.subplot(2,1,1)
    pylab.plot(np.real(u_anal[:,Ny/2,Nx/2]), c = col[1],  label="analy")
    pylab.plot(np.real(u_bpm[:,Ny/2,Nx/2]), c = col[0], label="bpm")
    pylab.legend()
    pylab.title("real")
    pylab.subplot(2,1,2)
    pylab.plot(np.imag(u_anal[:,Ny/2,Nx/2]), c = col[1],  label="analy")
    pylab.plot(np.imag(u_bpm[:,Ny/2,Nx/2]), c = col[0], label="bpm")
    pylab.legend()
    pylab.title("imag")



    pylab.show()
    pylab.draw()

