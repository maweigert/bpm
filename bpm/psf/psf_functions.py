"""

this is the main module defining 3d psf functions and initial focus fields


mweigert@mpi-cbg.de

"""


from bpm.psf._focus_fields_debye import focus_field_debye
from bpm.psf._focus_fields_cylindrical import focus_field_cylindrical

import numpy as np

__all__ =["psf_debye","psf_debye_u0","psf_lightsheet","psf_cylindrical","psf_cylindrical_u0"]

def psf_debye(shape,units,lam, NA, n0 = 1.,
              n_integration_steps = 200,
              return_field = False):
    """
    calculates the psf for a perfect, aberration free optical system
    via the vectorial debye diffraction integral

    see
    Matthew R. Foreman, Peter Toeroek,
    Computational methods in vectorial imaging,
    Journal of Modern Optics, 2011, 58, 5-6, 339


    returns:
    u, the (not normalized) intensity

    or if return_all_fields = True
    u,ex,ey,ez

    NA can be either a single number or an even length list of NAs (for bessel beams), e.g.
    NA = [.1,.2,.5,.6] lets light through the annulus .1<.2 and .5<.6


    """

    u, ex, ey, ez = focus_field_debye(shape = shape, units = units,
                                   lam = lam, NA = NA, n0 = n0,
                                   n_integration_steps = n_integration_steps)
    if return_field:
        return u,ex, ey, ez
    else:
        return u



def psf_lightsheet(shape,units,lam_illum,NA_illum, lam_detect, NA_detect, n0 = 1.,
              n_integration_steps = 200,
              return_field = False):
    """
    """

    u_detect= psf_debye(shape = shape, units = units,
                         lam = lam_detect,
                         NA = NA_detect,
                         n0 = n0,
                        n_integration_steps= n_integration_steps)



    u_illum= psf_debye(shape = shape[::-1],
                       units = units[::-1],
                         lam = lam_illum,
                         NA = NA_illum,
                         n0 = n0,
                        n_integration_steps= n_integration_steps)

    u_illum = u_illum.transpose((2,1,0))

    return u_detect*u_illum





def psf_debye_u0(shape,units,zfoc,lam,NA, n0, n_integration_steps = 200):
    """calculates initial plane u0 of a beam focused at zfoc
    shape = (Nx,Ny)
    units = (dx,dy)
    NAs = e.g. (0,.6)
    """

    u, ex, ey, ez = psf_debye((Nx,Ny,4),(dx,dy,zfoc/2.),
                              n0 = n0,
                              lam = lam,NA = NA,
                              n_integration_steps= n_integration_steps)
    # return ex[0,...]
    #FIXME
    return ex[0,...].conjugate()

def psf_cylindrical(shape,units,lam,NA, n0=1.,
                    return_field = False,
                    n_integration_steps = 100):
    """returns psf of cylindrical lerns with given NA
    """
    u, ex = focus_field_cylindrical(shape = shape, units = units,
                                   lam = lam, NA = NA, n0 = n0,
                                   n_integration_steps = n_integration_steps)


    if return_field:
        return u,ex
    else:
        return u


def psf_cylindrical_u0(shape,units,zfoc,lam,NA, n0=1.,  n_integration_steps = 200):
    """calculates initial plane u0 of a cylidrical lense beam focused at zfoc
    shape = (Nx,Ny)
    units = (dx,dy)
    NA = e.g. 0.6
    """

    Nx, Ny = shape
    dx, dy = units


    u , ex = psf_cylindrical(shape = (Nx,Ny,4),units = (dx,dy,2.*zfoc/3.),
                              lam = lam,NA = NA,n0=n0)

    # return ex
    return ex[0,...].conjugate()
