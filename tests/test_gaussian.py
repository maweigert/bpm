import numpy as np
from bpm import bpm_3d


def gaussian_tilt(x, y, z, lam=.5, NA=.4, tilt=0):
    """a gaussian beam with given waist and tilt
    """
    x, z = np.cos(tilt)*x-np.sin(tilt)*z, np.cos(tilt)*z+np.sin(tilt)*x
    r = np.sqrt(x**2+y**2)
    k = 2.*np.pi/lam

    # the waist parameter
    w0 = lam/np.pi/np.arcsin(NA)
    zr = np.pi*w0**2/lam
    w = w0*np.sqrt(1+(z/zr)**2)
    R = z*(1+(zr/z)**2)
    gouy = np.arctan(z/zr)
    return w0/w*np.exp(-r**2/w**2)*np.exp(-1.j*(k*z+k*r**2/2./R-gouy))


def test_gaussian(size, units, NA = .4, tilt =0.):
    """ propagates a gaussian wave freely to the center
    """

    Nx, Ny, Nz = size
    dx, dy, dz = units

    lam = .5

    x = dx*(np.arange(-Nx/2, Nx/2)+.5)
    y = dy*(np.arange(-Ny/2, Ny/2)+.5)
    z = dz*(np.arange(-Nz/2, Nz/2)+.5)

    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")

    u0 = gaussian_tilt(X,Y,Z,lam = lam, NA= NA, tilt = 0)

    u = bpm_3d(size, units=units, lam=lam,
               n0=n0,
               u0=u0[0],
               #use_fresnel_approx=True
               )

    return u, u0





if __name__=='__main__':
    Nx, Ny, Nz = (128,)*3

    dx, dy, dz = (.2,)*3


    n0 = 1.

    u_bpm, u_anal = test_gaussian((Nx, Ny, Nz), (dx, dy, dz), NA = .6, tilt = 0.4)

    print "L2 difference = %.10f"%np.mean(np.abs(u_bpm-u_anal)**2)

    # import pylab
    # import seaborn
    #
    # col = seaborn.color_palette()
    #
    # pylab.figure(1)
    # pylab.clf()
    # pylab.subplot(1, 3, 1)
    # pylab.imshow(np.abs(u_anal[:, Ny/2, :]), cmap="hot")
    # pylab.grid("off")
    # pylab.axis("off")
    # pylab.title("anal,  NA = %s, n0 = %.2f"%(NA, n0))
    #
    # pylab.subplot(1, 3, 2)
    # pylab.imshow(np.abs(u_bpm[:, Ny/2, :]), cmap="hot")
    # pylab.grid("off")
    # pylab.axis("off")
    # pylab.title("bpm,  NA = %s, n0 = %.2f"%(NA, n0))
    # pylab.subplot(1, 3, 3)
    # pylab.imshow(np.abs(u_bpm[:, Ny/2, :]-u_anal[:, Ny/2, :]), cmap="gray")
    # pylab.grid("off")
    # pylab.axis("off")
    # pylab.title("difference")
    #
    # pylab.figure(2)
    # pylab.clf()
    #
    # pylab.subplot(2, 1, 1)
    # pylab.plot(np.real(u_anal[:, Ny/2, Nx/2]), c=col[1], label="analy")
    # pylab.plot(np.real(u_bpm[:, Ny/2, Nx/2]), c=col[0], label="bpm")
    # pylab.legend()
    # pylab.title("real")
    # pylab.subplot(2, 1, 2)
    # pylab.plot(np.imag(u_anal[:, Ny/2, Nx/2]), c=col[1], label="analy")
    # pylab.plot(np.imag(u_bpm[:, Ny/2, Nx/2]), c=col[0], label="bpm")
    # pylab.legend()
    # pylab.title("imag")
    #
    # pylab.show()
    # pylab.draw()
