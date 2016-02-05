"""

test whether the psf calculations are correct

"""

import numpy as np
import numpy.testing as npt
from scipy.special import jn

from bpm import psf_debye, psf_lightsheet
from spimagine import read3dTiff

import pylab


def theory_x(r,lam,NA,n):
    """theoretical x profile in the scalar limit"""
    v = 2.*np.pi*NA/lam*r
    res =  jn(1,v)**2/v**2
    res[np.isnan(res)] = .25
    res *= 1./np.amax(res)
    return res

def theory_debye_x(r,lam,NA,n):
    """theoretical x profile via debye integral"""
    v = 2.*np.pi*NA/lam*r
    res =  jn(1,v)**2/v**2
    res[np.isnan(res)] = .25
    res *= 1./np.amax(res)
    return res

def theory_z(z,lam,NA,n):
    """zprofile"""
    u = 2.*np.pi/lam*NA**2/n*z
    res =  16./u**2*np.sin(u/4.)**2
    res[np.isnan(res)] = 1.
    res *= 1./np.amax(res)
    return res

def theory_debye_z(z,lam,NA,n):
    """zprofile"""
    nint = 1000
    a = np.arcsin(NA)
    t = np.linspace(0.,a,nint)

    def part_sum(z0):
        co = np.cos(t)
        si = np.sin(t)
        integ = np.sqrt(co)*si*(co+1)*np.exp(1.j*2*np.pi/lam*z0*co/n)

        #integ = t*np.exp(-1.j*2*np.pi/lam*z0*t**2/2.)
        res = abs((t[1]-t[0])*np.sum(integ))**2
        return res

    res =  np.array([part_sum(_z) for _z in z])

    res *= 1./np.amax(res)
    return res


def compare(fname,NA,dx,lam,n):

    u1 = read3dTiff(fname)

    Nz,Ny,Nx = u1.shape
    u2 = psf_debye((Nx, Ny, Nz), (dx,) * 3, lam, NA, n0=n)

    u1 *= 1./np.amax(u1)
    u2 *= 1./np.amax(u2)

    x = dx*(np.arange(Nx)-Nx/2)
    z = dx*(np.arange(Nz)-Nz/2)


    pylab.plot(x,u1[Nz/2,Ny/2,:],label="u1_x")
    pylab.plot(x,u2[Nz/2,Ny/2,:],label="u2_x")
    pylab.plot(x,theory_x(x,lam,NA,n),"o",color="k",label="theor_x")

    pylab.plot(z,u1[:,Ny/2,Nx/2],label="u1_z")
    pylab.plot(z,u2[:,Ny/2,Nx/2],label="u2_z")
    pylab.plot(z,theory_z(z,lam,NA,n),"o",color="k",label="theor_z")
    pylab.plot(z,theory_debye_z(z,lam,NA,n),"o",color="r",label="debye_z")
    return u1, u2

def compare_theory(shape, NA,dx,lam,n):

    Nz,Ny,Nx = shape
    u2 = psf_debye((Nx, Ny, Nz), (dx,) * 3, lam, NA, n0=n)

    u2 *= 1./np.amax(u2)

    x = dx*(np.arange(Nx)-Nx/2)
    z = dx*(np.arange(Nz)-Nz/2)

    # pylab.plot(x,u2[Nz/2,Ny/2,:],label="u2_x")
    # pylab.plot(x,theory_x(x,lam,NA,n),color="k",label="theor_x")

    pylab.plot(z,u2[:,Ny/2,Nx/2],label="u2_z")
    pylab.plot(z,theory_z(z,lam,NA,n),label="theor_z")
    pylab.plot(z,theory_debye_z(z,lam,NA,n),label="debye_z")
    return u2


def test_lightsheet(NA_illum, NA_detect, n0=1.):
    lam_illum  = .4
    lam_detect = .5
    N = 128
    dx = .05
    u = psf_lightsheet((N,)*3,(dx,)*3,lam_illum,NA_illum,lam_detect,NA_detect, n0=n0)
    return u



if __name__ == '__main__':

    h = test_lightsheet(.15,.2, n0= 1.33)

    pylab.figure(1)
    pylab.clf()
    pylab.subplot(1,2,1)
    pylab.imshow(h[h.shape[0]/2,...],cmap="hot")
    pylab.axis("off")
    pylab.subplot(1,2,2)
    pylab.imshow(h[...,h.shape[-1]/2],cmap="hot")
    pylab.axis("off")
    pylab.show()

    #
    # pylab.figure(1)
    # pylab.clf()
    # NA = .6
    # u2 = compare_theory((128,)*3, NA,.05,.5,1.5)
    # pylab.legend()
    # pylab.title("NA = %s"%NA)
    # pylab.show()
    #
    # pylab.figure(2)
    # pylab.clf()
    # NA = .9
    # u2 = compare_theory((128,)*3, NA,.05,.5,1.5)
    # pylab.legend()
    # pylab.title("NA = %s"%NA)
    # pylab.show()


 # pylab.figure(1)
    # pylab.clf()
    # u1, u2 = compare("data/psf_NA_0.8_dx_0.1_n_1.0.tif", .8,.05,.5,1.)
    # pylab.legend()
    # pylab.show()

    # pylab.figure(2)
    # pylab.clf()
    # u1, u2 = compare("data/psf_NA_0.8_dx_0.1_n_1.5.tif", .8,.05,.5,1.5)
    # pylab.legend()
    # pylab.show()