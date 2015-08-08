"""
PSF calulcations for high NA opbejctives via the Debye Wolf integral


see e.g.
Foreman, M. R., & Toeroek, P. (2011). Computational methods in vectorial imaging.
Journal of Modern Optics, 58(5-6)
"""


from volust.volgpu import *
from volust.volgpu.oclalgos import *

import itertools

import numpy as np

import time


import os
import sys

def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


def test_bessel(n,x):
    x_g = OCLArray.from_array(x.astype(float32))
    res_g = OCLArray.empty_like(x.astype(float32))
    
    p = OCLProgram(absPath("bessel.cl"))
    p.run_kernel("bessel_fill",x_g.shape,None,
                 x_g.data,res_g.data,int32(n))

    return res_g.get()


def psf_debye(shape,units,lam,NAs, n_integration_steps = 200):
    """NAs is an increasing list of NAs
    NAs = [.1,.2,.5,.6] lets light through the annulus .1<.2 and .5<.6
    """

    p = OCLProgram(absPath("psf_debye.cl"),build_options = str("-I %s -D INT_STEPS=%s"%(absPath("."),n_integration_steps)))

    
    Nx0, Ny0, Nz0 = shape
    dx, dy, dz = units

    alphas = np.arcsin(np.array(NAs))
    
    Nx = (Nx0+1)/2
    Ny = (Ny0+1)/2
    Nz = (Nz0+1)/2

    u_g = OCLArray.empty((Nz,Ny,Nx),np.float32)
    ex_g = OCLArray.empty(u_g.shape,np.complex64)
    ey_g = OCLArray.empty(u_g.shape,np.complex64)
    ez_g = OCLArray.empty(u_g.shape,np.complex64)

    alpha_g = OCLArray.from_array(alphas.astype(np.float32))

    t = time.time()
    
    p.run_kernel("debye_wolf",u_g.shape[::-1],None,
                 ex_g.data,ey_g.data,ez_g.data, u_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(0),np.float32(dx*Nx),
                 np.float32(0),np.float32(dy*Ny),
                 np.float32(0),np.float32(dz*Nz),                 
                 np.float32(lam),
                 alpha_g.data, np.int32(len(alphas)))

    u = u_g.get()
    ex = ex_g.get()
    ey = ey_g.get()
    ez = ez_g.get()

    print "time in secs:" , time.time()-t
    
    u_all = np.empty((Nz0,Ny0,Nx0),np.float32)
    ex_all = np.empty((Nz0,Ny0,Nx0),np.complex64)
    ey_all = np.empty((Nz0,Ny0,Nx0),np.complex64)
    ez_all = np.empty((Nz0,Ny0,Nx0),np.complex64)

    sx = [slice(0,Nx),slice(Nx0-Nx0/2,Nx0)]
    sy = [slice(0,Ny),slice(Ny0-Ny0/2,Ny0)]
    sz = [slice(0,Nz),slice(Nz0-Nz0/2,Nz0)]

    sx = [slice(0,Nx),slice(Nx0-Nx,Nx0)]
    sy = [slice(0,Ny),slice(Ny0-Ny,Ny0)]
    sz = [slice(0,Nz),slice(Nz0-Nz,Nz0)]

    for i,j,k in itertools.product([0,1],[0,1],[0,1]):
        u_all[sz[1-i],sy[1-j],sx[1-k]] = u[::(-1)**i,::(-1)**j,::(-1)**k]
        ex_all[sz[1-i],sy[1-j],sx[1-k]] = ex[::(-1)**i,::(-1)**j,::(-1)**k]
        ey_all[sz[1-i],sy[1-j],sx[1-k]] = ey[::(-1)**i,::(-1)**j,::(-1)**k]
        ez_all[sz[1-i],sy[1-j],sx[1-k]] = ez[::(-1)**i,::(-1)**j,::(-1)**k]

        
    return u_all, ex_all, ey_all, ez_all



def psf_debye_new(shape,units,lam,NAs, n_integration_steps = 200):
    """NAs is an increasing list of NAs
    NAs = [.1,.2,.5,.6] lets light through the annulus .1<.2 and .5<.6
    """
    
    p = OCLProgram(absPath("psf_debye.cl"),
                   build_options = str("-I %s -D INT_STEPS=%s"%(absPath("."),n_integration_steps)))

    
    Nx, Ny, Nz = shape
    dx, dy, dz = units

    print dx,dy,dz
    alphas = np.arcsin(np.array(NAs))

    print alphas
    Nrad = max(Nx/2,Ny/2)

    Rmax = .5*np.sqrt(dx**2*Nx**2+Ny**2*dy**2)
    Zmax = .5*dz*Nz

    # the values of I_0, I_1... as a function of r and z
    I_vals_re = OCLImage.empty((Nrad,Nz/2),np.dtype(np.float32),num_channels = 4)
    I_vals_im = OCLImage.empty((Nrad,Nz/2),np.dtype(np.float32),num_channels = 4)

    u_g = OCLArray.empty((Nz,Ny,Nx),np.float32)
    ex_g = OCLArray.empty(u_g.shape,np.complex64)
    ey_g = OCLArray.empty(u_g.shape,np.complex64)
    ez_g = OCLArray.empty(u_g.shape,np.complex64)

    alpha_g = OCLArray.from_array(alphas.astype(np.float32))

    t = time.time()

        
    p.run_kernel("precalculate_I",I_vals_im.shape,None,
                 I_vals_re,
                 I_vals_im,
                 np.float32(lam),
                 np.float32(Rmax),
                 np.float32(Zmax),
                 alpha_g.data,
                 np.int32(len(alphas)))

    # return I_vals_re.get(),I_vals_im.get(),1,1

    p.run_kernel("assemble_I",u_g.shape[::-1],None,
                 I_vals_re,I_vals_im,
                 ex_g.data,ey_g.data,ez_g.data, u_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(Rmax),
                 np.float32(Zmax),
                 np.float32(-dx*(Nx-1)/2.),np.float32(dx*(Nx-1)/2.),
                 np.float32(-dy*(Ny-1)/2.),np.float32(dy*(Ny-1)/2.),
                 np.float32(-dz*(Nz-1)/2.),np.float32(dz*(Nz-1)/2.))

    return u_g.get(), ex_g.get(), ey_g.get(), ez_g.get()

def psf_debye_mask(shape,units,lam, n_integration_steps = 200):
    
    p = OCLProgram(absPath("psf_debye_fullmask.cl"),
                   build_options = str("-I %s -D INT_STEPS=%s"%(absPath("."),n_integration_steps)))

    
    Nx, Ny, Nz = shape
    dx, dy, dz = units

    Nrad = max(Nx/2,Ny/2)

    Rmax = .5*np.sqrt(dx**2*Nx**2+Ny**2*dy**2)
    Zmax = .5*dz*Nz

    # the values of I_0, I_1... as a function of r and z
    I_vals_re = OCLImage.empty((Nrad,Nz/2),np.dtype(np.float32),num_channels = 4)
    I_vals_im = OCLImage.empty((Nrad,Nz/2),np.dtype(np.float32),num_channels = 4)

    u_g = OCLArray.empty((Nz,Ny,Nx),np.float32)
    ex_g = OCLArray.empty(u_g.shape,np.complex64)
    ey_g = OCLArray.empty(u_g.shape,np.complex64)
    ez_g = OCLArray.empty(u_g.shape,np.complex64)

    t = time.time()

        
    p.run_kernel("precalculate_I",I_vals_im.shape,None,
                 I_vals_re,
                 I_vals_im,
                 np.float32(lam),
                 np.float32(Rmax),
                 np.float32(Zmax))

    # return I_vals_re.get(),I_vals_im.get(),1,1

    p.run_kernel("assemble_I",u_g.shape[::-1],None,
                 I_vals_re,I_vals_im,
                 ex_g.data,ey_g.data,ez_g.data, u_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(Rmax),
                 np.float32(Zmax),
                 np.float32(-dx*(Nx-1)/2.),np.float32(dx*(Nx-1)/2.),
                 np.float32(-dy*(Ny-1)/2.),np.float32(dy*(Ny-1)/2.),
                 np.float32(-dz*(Nz-1)/2.),np.float32(dz*(Nz-1)/2.))

    return u_g.get(), ex_g.get(), ey_g.get(), ez_g.get()

def psf_debye_slit(shape,units,lam,NAs, slit_xs, slit_sigmas,
                   n_integration_steps = 100):
    """NAs is an increasing list of NAs
    NAs = [.1,.2,.5,.6] lets light through the annulus .1<.2 and .5<.6

    slit_x are the x coordinates pf the slits
    """
    
    p = OCLProgram(absPath("psf_debye.cl"),
                   build_options = "-I %s -D INT_STEPS=%s"%(absPath("."),n_integration_steps))

    
    Nx0, Ny0, Nz0 = shape
    dx, dy, dz = units

    alphas = np.arcsin(np.array(NAs))
    
    Nx = (Nx0+1)/2
    Ny = (Ny0+1)/2
    Nz = (Nz0+1)/2

    u_g = OCLArray.empty((Nz,Ny,Nx),np.float32)
    ex_g = OCLArray.empty(u_g.shape,np.complex64)
    ey_g = OCLArray.empty(u_g.shape,np.complex64)
    ez_g = OCLArray.empty(u_g.shape,np.complex64)

    alpha_g = OCLArray.from_array(alphas.astype(np.float32))

    slit_xs_g = OCLArray.from_array(np.array(slit_xs).astype(np.float32))
    slit_sigmas_g = OCLArray.from_array(np.array(slit_sigmas).astype(np.float32))

    t = time.time()
    
    p.run_kernel("debye_wolf_slit",u_g.shape[::-1],None,
                 ex_g.data,ey_g.data,ez_g.data, u_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(0),np.float32(dx*Nx),
                 np.float32(0),np.float32(dy*Ny),
                 np.float32(0),np.float32(dz*Nz),                 
                 np.float32(lam),
                 alpha_g.data, np.int32(len(alphas)),
                 slit_xs_g.data, slit_sigmas_g.data, np.int32(len(slit_xs)))

    u = u_g.get()
    ex = ex_g.get()
    ey = ey_g.get()
    ez = ez_g.get()

    print "time in secs:" , time.time()-t
    
    u_all = np.empty((Nz0,Ny0,Nx0),np.float32)
    ex_all = np.empty((Nz0,Ny0,Nx0),np.complex64)
    ey_all = np.empty((Nz0,Ny0,Nx0),np.complex64)
    ez_all = np.empty((Nz0,Ny0,Nx0),np.complex64)

    sx = [slice(0,Nx),slice(Nx0-Nx0/2,Nx0)]
    sy = [slice(0,Ny),slice(Ny0-Ny0/2,Ny0)]
    sz = [slice(0,Nz),slice(Nz0-Nz0/2,Nz0)]

    sx = [slice(0,Nx),slice(Nx0-Nx,Nx0)]
    sy = [slice(0,Ny),slice(Ny0-Ny,Ny0)]
    sz = [slice(0,Nz),slice(Nz0-Nz,Nz0)]

    for i,j,k in itertools.product([0,1],[0,1],[0,1]):
        u_all[sz[1-i],sy[1-j],sx[1-k]] = u[::(-1)**i,::(-1)**j,::(-1)**k]
        ex_all[sz[1-i],sy[1-j],sx[1-k]] = ex[::(-1)**i,::(-1)**j,::(-1)**k]
        ey_all[sz[1-i],sy[1-j],sx[1-k]] = ey[::(-1)**i,::(-1)**j,::(-1)**k]
        ez_all[sz[1-i],sy[1-j],sx[1-k]] = ez[::(-1)**i,::(-1)**j,::(-1)**k]

        
    return u_all, ex_all, ey_all, ez_all


def psf_focus_u0(shape,units,zfoc,lam,NAs, n_integration_steps = 200):
    """calculates initial plane u0 of a beam focused at zfoc
    shape = (Nx,Ny)
    units = (dx,dy)
    NAs = e.g. (0,.6)
    """

    if not isinstance(NAs,(list,tuple)):
        NAs = [0,NAs]
        
    Nx, Ny = shape
    dx, dy = units
    h, ex, ey, ez = psf_debye((Nx,Ny,4),(dx,dy,zfoc/2.),
                              lam = lam,NAs = NAs)
    return ex[0,...].conjugate()



    
def test_debye():
    
    lam = .5
    NA1 = .7
    NA2 = .76

    Nx = 128
    Nz = 128 
    dx = .05
    
    u,ex,ey,ez = psf_debye((Nx,Nx,Nz),(dx,dx,dx),
                           lam = lam, NAs = [0.89,0.9], n_integration_steps= 100)

    u2,ex2,ey2,ez2 = psf_debye_new((Nx,Nx,Nz),(dx,dx,dx),
                           lam = lam, NAs = [0.89,0.9], n_integration_steps= 100)
    
    return u,u2

if __name__ == '__main__':

    # u,u2 = test_debye() 


    u,ex,ey,ez = psf_debye_mask((128,)*3,(.05,)*3,
                           lam = .5, n_integration_steps= 100)
