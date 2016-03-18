"""

test whether the psf calculations are correct

"""

import numpy as np
import numpy.testing as npt
from scipy.special import jn

from bpm import psf, psf_lightsheet
from spimagine import read3dTiff

import pylab


def theory_x(r,lam,NA,n):
    """theoretical x profile in the scalar limit"""
    v = 2.*np.pi*NA/lam*r
    res =  jn(1,v)**2/v**2
    res[np.isnan(res)] = .25
    res *= 1./np.amax(res)
    return res


def theory_z(z,lam,NA,n):
    """theoretical z profile in the scalar limit"""
    u = 2.*np.pi/lam*NA**2/n*z
    res =  16./u**2*np.sin(u/4.)**2
    res[np.isnan(res)] = 1.
    res *= 1./np.amax(res)
    return res

def theory_debye_z(z,lam,NA,n):
    """theoretical z profile via debye integral"""
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


def compare_with_file(fname,dx, NA,lam,n):

    u1 = read3dTiff(fname)

    Nz,Ny,Nx = u1.shape
    u2 = psf((Nx, Ny, Nz), (dx,) * 3, lam, NA, n0=n)

    u1 *= 1./np.amax(u1)
    u2 *= 1./np.amax(u2)

    x = dx*(np.arange(Nx)-Nx/2)
    z = dx*(np.arange(Nz)-Nz/2)

    pylab.clf()
    pylab.subplot(2,1,1)
    pylab.title("NA = %s n = %s"%(NA,n))
    pylab.plot(x,u1[Nz/2,Ny/2,:],label="file_x")
    pylab.plot(x,u2[Nz/2,Ny/2,:],label="u_x")
    pylab.plot(x,theory_x(x,lam,NA,n),color="k",label="theor_x")
    pylab.legend()

    pylab.subplot(2,1,2)

    pylab.plot(z,u1[:,Ny/2,Nx/2],label="file_z")
    pylab.plot(z,u2[:,Ny/2,Nx/2],label="u_z")
    pylab.plot(z,theory_z(z,lam,NA,n),color="k",label="theor_z")
    #pylab.plot(z,theory_debye_z(z,lam,NA,n),"o",color="r",label="debye_z")
    pylab.legend()
    pylab.show()
    return u1, u2

def compare_with_theory(shape, dx, NA,lam,n):

    Nz,Ny,Nx = shape
    u2 = psf((Nx, Ny, Nz), (dx,) * 3, lam, NA, n0=n)

    u2 *= 1./np.amax(u2)

    x = dx*(np.arange(Nx)-Nx/2)
    z = dx*(np.arange(Nz)-Nz/2)

    # pylab.plot(x,u2[Nz/2,Ny/2,:],label="u2_x")
    # pylab.plot(x,theory_x(x,lam,NA,n),color="k",label="theor_x")

    #pylab.figure()
    pylab.clf()
    pylab.subplot(2,1,1)

    pylab.plot(x,u2[Nz/2,Ny/2,:],label="u2_x")
    pylab.plot(x,theory_x(x,lam,NA,n),label="theor_x")
    #pylab.plot(x,theory_debye_x(x,lam,NA,n),label="debye_x")
    pylab.title("NA = %s n = %s"%(NA,n))
    pylab.legend()
    pylab.subplot(2,1,2)
    pylab.plot(z,u2[:,Ny/2,Nx/2],label="u2_z")
    pylab.plot(z,theory_z(z,lam,NA,n),label="theor_z")
    pylab.plot(z,theory_debye_z(z,lam,NA,n),label="debye_z")
    pylab.legend()
    pylab.show()
    return u2

def _get_error(shape, dx, NA,lam,n):
    Nz,Ny,Nx = shape
    u2 = psf((Nx, Ny, Nz), (dx,) * 3, lam, NA, n0=n)
    u2 *= 1./np.amax(u2)
    x = dx*(np.arange(Nx)-Nx/2)
    z = dx*(np.arange(Nz)-Nz/2)

    u2_z = u2[:,Ny/2,Nx/2]
    u_z = theory_z(z,lam,NA,n)
    u2_x = u2[Nz/2,Ny/2,:]
    u_x = theory_x(x,lam,NA,n)

    return np.sqrt(np.mean((u_x-u2_x)**2))+np.sqrt(np.mean((u_z-u2_z)**2))

def compare_error(shape, dx, NAs,lam,n):
    errs = [_get_error(shape,dx,NA,lam,n) for NA in NAs]
    pylab.figure()
    pylab.plot(NAs,errs,label = "error")
    pylab.xlabel("NA")
    pylab.ylabel("RMS")
    pylab.show()

def test_lightsheet(NA_illum, NA_detect, n0=1.):
    lam_illum  = .4
    lam_detect = .5
    N = 128
    dx = .05
    u = psf_lightsheet((N,)*3,(dx,)*3,lam_illum,NA_illum,lam_detect,NA_detect, n0=n0)
    return u



if __name__ == '__main__':

    pylab.figure()
    compare_with_file("data/psf_NA_0.8_dx_0.1_n_1.0.tif",
                      dx = 0.05, NA = .8,lam = .5, n= 1.)
    pylab.figure()
    compare_with_file("data/psf_NA_0.8_dx_0.1_n_1.5.tif",
                      dx = 0.05, NA = .8,lam = .5, n= 1.5)

    pylab.figure()
    compare_with_theory((256,64,64),dx = .05,NA = 0.2, lam = .5, n=1.0)
    pylab.figure()
    compare_with_theory((256,64,64),dx = .05,NA = 0.2, lam = .5, n=1.3)
    pylab.figure()
    compare_with_theory((256,64,64),dx = .05,NA = 0.2, lam = .5, n=1.5)

    pylab.ioff()


    # compare_error((256,64,64),dx = .05,NAs = np.linspace(.1,.8,10),
    #               lam = .5, n=1.)
    # #
