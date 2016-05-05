"""

test whether the psf calculations are correct

"""

import numpy as np
from scipy.special import jn
from bpm import psf


import pylab


def debye_integral_at_point(x,y,z, k,NA, n_int = 1000):
    from scipy.special import jn

    a = np.arcsin(NA)

    kr = k*np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    kz = k*z

    t = np.linspace(0.,a,n_int)
    dt = a/(n_int-1.)

    f_0 = np.sqrt(np.cos(t))*np.sin(t)*(np.cos(t)+1.)*jn(0,kr*np.sin(t))*np.exp(1.j*kz*np.cos(t))
    f_1 = np.sqrt(np.cos(t))*np.sin(t)**2*jn(1,kr*np.sin(t))*np.exp(1.j*kz*np.cos(t))
    f_2 = np.sqrt(np.cos(t))*np.sin(t)*(np.cos(t)-1.)*jn(2,kr*np.sin(t))*np.exp(1.j*kz*np.cos(t))

    I0 = dt*(np.sum(f_0)-.5*(f_0[0]+f_0[-1]))
    I1 = dt*(np.sum(f_1)-.5*(f_1[0]+f_1[-1]))
    I2 = dt*(np.sum(f_2)-.5*(f_2[0]+f_2[-1]))
    

    ex = I0 +I2*np.cos(2*phi)
    ey = I2*np.sin(2*phi)
    ez = -2.j*I1*np.cos(phi)

    u = abs(ex)**2+abs(ey)**2+abs(ez)**2

    return u,ex,ey,ez


if __name__ == '__main__':

    NA = .95
    lam = .5


    Nx = 256
    Ny = 164
    Nz = 100
    dx = 0.008
    u, ex,ey,ez = psf((Nx,Ny,Nz),(dx,)*3,lam=lam, NA = NA, return_field=True)
    #
    x = dx*(np.arange(Nx)-Nx/2.)
    y = dx*(np.arange(Ny)-Ny/2.)


    ex0 = np.array([debye_integral_at_point(_x,.5*dx,.5*dx, 2*np.pi/lam,NA,
                                            n_int = 1000)[1] for _x in x])

    ex1 = np.array([debye_integral_at_point(.5*dx,_x,.5*dx, 2*np.pi/lam,NA,
                                            n_int = 1000)[1] for _x in x])



    #ex1 = ex[Nz/2,Ny/2,:]

    #
    # pylab.clf()
    # for i,comp in enumerate([ex,ey,ez]):
    #     pylab.subplot(2,2,i+1)
    #     pylab.imshow(np.abs(comp[Nz/2,...]),cmap = "jet")
    #     pylab.xlabel("x")
    #     pylab.ylabel("y")
    #
    #     pylab.colorbar()
    #
    #
    #
    #
    # pylab.subplot(2,2,4)
    # pylab.imshow(u[Nz/2,...],cmap = "jet")
    #
    # pylab.colorbar()
