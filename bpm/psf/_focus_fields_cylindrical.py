# coding=utf-8
"""
PSF calulcations for cylidrical lenses

see e.g.
Purnapatra, Subhajit B. Mondal, Partha P.
Determination of electric field at and near the focus of a cylindrical lens for applications in fluorescence microscopy (2013)
"""


from gputools import OCLArray, OCLImage, OCLProgram

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



def focus_field_cylindrical(shape,units,lam = .5,NA = .3, n0=1.,
                            n_integration_steps = 100):
    """computes focus field of cylindrical lerns with given NA

    see:
    Colin J. R. Sheppard,
    Cylindrical lenses—focusing and imaging: a review

    Appl. Opt. 52, 538-545 (2013)

    return u,ex,ey,ez   with u being the intensity
    """

    p = OCLProgram(absPath("kernels/psf_cylindrical.cl"),build_options = str("-I %s -D INT_STEPS=%s"%(absPath("."),n_integration_steps)))

    
    Nx, Ny, Nz = shape
    dx, dy, dz = units

    alpha = np.arcsin(NA/n0)
    
    u_g = OCLArray.empty((Nz,Ny),np.float32)
    ex_g = OCLArray.empty((Nz,Ny),np.complex64)
    ey_g = OCLArray.empty((Nz,Ny),np.complex64)
    ez_g = OCLArray.empty((Nz,Ny),np.complex64)

    t = time.time()
    
    p.run_kernel("psf_cylindrical",u_g.shape[::-1],None,
                 ex_g.data,
                 ey_g.data,
                 ez_g.data,
                 u_g.data,
                 np.float32(-dy*(Ny-1)/2.),np.float32(dy*(Ny-1)/2.),
                 np.float32(-dz*(Nz-1)/2.),np.float32(dz*(Nz-1)/2.),
                 np.float32(lam/n0),
                 np.float32(alpha))

    u = np.array(np.repeat(u_g.get()[...,np.newaxis],Nx,axis=-1))
    ex = np.array(np.repeat(ex_g.get()[...,np.newaxis],Nx,axis=-1))
    ey = np.array(np.repeat(ey_g.get()[...,np.newaxis],Nx,axis=-1))
    ez = np.array(np.repeat(ez_g.get()[...,np.newaxis],Nx,axis=-1))

    
    print "time in secs:" , time.time()-t
    

    return u, ex, ey, ez



def focus_field_cylindrical_plane(shape = (128,128),
                            units = (.1,.1),
                            z = 0.,
                            lam = .5, NA = .6, n0 = 1.,
                            ex_g = None,
                            n_integration_steps = 200):
    """
    calculates the x component of the electric field  at a given z position z for a perfect, aberration free optical system
    via the vectorial debye diffraction integral for a cylindrical lens

    see
    Colin J. R. Sheppard,
    Cylindrical lenses—focusing and imaging: a review

    Appl. Opt. 52, 538-545 (2013)


    if ex_g is a valid OCLArray it fills it and returns None
    otherwise returns ex as a numpy array


    """

    p = OCLProgram(absPath("kernels/psf_cylindrical.cl"),build_options = str("-I %s -D INT_STEPS=%s"%(absPath("."),n_integration_steps)))


    Nx, Ny = shape
    dx, dy = units

    alpha = np.arcsin(NA/n0)

    if ex_g is None:
        use_buffer = False
        ex_g = OCLArray.empty((Ny,Nx),np.complex64)
    else:
        use_buffer = True

    assert ex_g.shape[::-1] == shape


    p.run_kernel("psf_cylindrical_plane",(Nx,Ny),None,
                 ex_g.data,
                 np.float32(-dy*(Ny-1)/2.),np.float32(dy*(Ny-1)/2.),
                 np.float32(z),
                 np.float32(lam/n0),
                 np.float32(alpha))

    if not use_buffer:
        return ex_g.get()



if __name__ == '__main__':

    u, ex, ey, ez = focus_field_cylindrical((128,)*3,(.1,)*3,NA = .3)


    #ex2  = focus_field_cylindrical_plane((128,)*2,(.1,)*2,z = -6.4, NA = .01)
