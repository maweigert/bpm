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


def poly_points(N=6):
    ts = np.pi/2+np.linspace(0,2*np.pi,N+1)[:-1]
    return np.stack([np.cos(ts),np.sin(ts)])


def focus_field_lattice(shape,units,lam = .5,NA1=.4,NA2=.5,
                        sigma = .1,
                        Npoly = 6,
                        n0=1., n_integration_steps = 100):
    """
    """


    kxs, kys = .5*(NA1+NA2)*poly_points(Npoly)

    p = OCLProgram(absPath("kernels/psf_lattice.cl"),
            build_options = ["-I",absPath("kernels"),"-D","INT_STEPS=%s"%n_integration_steps])


    kxs = np.array(kxs)
    kys = np.array(kys)

    Nx, Ny, Nz = shape
    dx, dy, dz = units

    alpha1 = np.arcsin(NA1/n0)
    alpha2 = np.arcsin(NA2/n0)


    u_g = OCLArray.empty((Nz,Ny,Nx),np.float32)
    ex_g = OCLArray.empty((Nz,Ny,Nx),np.complex64)
    ey_g = OCLArray.empty((Nz,Ny,Nx),np.complex64)
    ez_g = OCLArray.empty((Nz,Ny,Nx),np.complex64)


    kxs_g = OCLArray.from_array(kxs.astype(np.float32))
    kys_g = OCLArray.from_array(kys.astype(np.float32))

    t = time.time()


    
    p.run_kernel("debye_wolf_lattice",(Nx,Ny,Nz),
                 None,
                 ex_g.data,
                 ey_g.data,
                 ez_g.data,
                 u_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(-dx*(Nx-1)/2.),np.float32(dx*(Nx-1)/2.),
                 np.float32(-dy*(Ny-1)/2.),np.float32(dy*(Ny-1)/2.),
                 np.float32(-dz*(Nz-1)/2.),np.float32(dz*(Nz-1)/2.),
                 np.float32(1.*lam/n0),
                 np.float32(alpha1),
                 np.float32(alpha2),
                 kxs_g.data,
                 kys_g.data,
                 np.int32(len(kxs)),
                 np.float32(sigma)
    )

    ex = ex_g.get()


    
    print "time in secs:" , time.time()-t
    return ex

def focus_field_lattice_plane(shape = (256,256),
                              units=(.1,.1),
                              zpos = 0.,
                              lam = .5,
                              NA1=.4,NA2=.5,
                              sigma = .1,
                              Npoly = 6,
                              n0 = 1.,
                              apodization_bound = 10,
                              ex_g = None,
                              n_integration_steps = 100):
    """
    """




    p = OCLProgram(absPath("kernels/psf_lattice.cl"),
            build_options = ["-I",absPath("kernels"),"-D","INT_STEPS=%s"%n_integration_steps])



    Nx, Ny = shape
    dx, dy = units

    alpha1 = np.arcsin(NA1/n0)
    alpha2 = np.arcsin(NA2/n0)

    kxs, kys = .5*(alpha1+alpha2)*poly_points(Npoly)

    if ex_g is None:
        use_buffer = False
        ex_g = OCLArray.empty((Ny,Nx),np.complex64)
    else:
        use_buffer = True

    assert ex_g.shape[::-1] == shape


    kxs_g = OCLArray.from_array(kxs.astype(np.float32))
    kys_g = OCLArray.from_array(kys.astype(np.float32))

    t = time.time()



    p.run_kernel("debye_wolf_lattice_plane",(Nx,Ny),
                 None,
                 ex_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(-dx*(Nx-1)/2.),np.float32(dx*(Nx-1)/2.),
                 np.float32(-dy*(Ny-1)/2.),np.float32(dy*(Ny-1)/2.),
                 np.float32(zpos),
                 np.float32(1.*lam/n0),
                 np.float32(alpha1),
                 np.float32(alpha2),
                 kxs_g.data,
                 kys_g.data,
                 np.int32(len(kxs)),
                 np.float32(sigma),
                 np.int32(apodization_bound),
    )




    if not use_buffer:
        res =  ex_g.get()
        print "time in secs:" , time.time()-t
        return res



if __name__ == '__main__':

    # ex = focus_field_lattice_plane((128,128),(.1,)*2,zpos = -12.8*0,
    #                          NA1 = .44 ,NA2 =.55,
    #                          sigma = .1,
    #                          Npoly = 6,
    #                         lam = .488,
    #                          n0= 1.33,
    #                         apodization_bound=0,
    #                          n_integration_steps=500)


    ex = focus_field_lattice_plane((512,256),(.1,)*2,
                                   zpos = -12.8,
                             NA1 = .44 ,NA2 =.55,
                             sigma = .1,
                             Npoly = 6,
                            lam = .5,
                             n0= 1.33,
                            apodization_bound=100,
                             n_integration_steps=500)
