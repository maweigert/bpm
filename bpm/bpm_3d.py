"""the main method for beam propagation in refractive media"""

import numpy as np

from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan
from gputools import OCLReductionKernel

from bpm.utils import StopWatch, absPath

from bpm._bpm_3d_buffer import _bpm_3d_buffer as _bpm_3d
from bpm._bpm_3d_buffer import _bpm_3d_buffer_split as _bpm_3d_split
from bpm._bpm_3d_buffer import _bpm_3d_buffer_free as _bpm_3d_free

from scipy.ndimage.interpolation import zoom

#this is the main method to calculate everything

def bpm_3d(size,
           units,
           lam = .5,
           u0 = None, dn = None,
           n_volumes = 1,
           n0 = 1.,
           return_scattering = False,
           return_g = False,
           use_fresnel_approx = False):
    """
    simulates the propagation of monochromativ wave of wavelength lam with initial conditions u0 along z in a media filled with dn

    size     -    the dimension of the image to be calulcated  in pixels (Nx,Ny,Nz)
    units    -    the unit lengths of each dimensions in microns
    lam      -    the wavelength
    u0       -    the initial field distribution, if u0 = None an incident
    plane wave is assumed
    dn       -    the refractive index of the medium (can be complex)
    n0       -    refractive index of surrounding medium
    """

    if n_volumes ==1:
        return _bpm_3d(size, units,
                       lam = lam,
                       u0 = u0, dn = dn,
                       n0 = n0,
                       return_scattering = return_scattering,
                       return_g = return_g,
                       use_fresnel_approx = use_fresnel_approx)
    else:
        return _bpm_3d_split(size, units,
                             lam = lam,
                             u0 = u0, dn = dn,
                             n_volumes = n_volumes,
                             n0 = n0,                            
                             return_scattering = return_scattering,
                             return_g = return_g,
                             use_fresnel_approx = use_fresnel_approx)

    
def bpm_3d_free(size, units, dz, lam = .5,
                u0 = None,
                n0 = 1., 
                use_fresnel_approx = False):
    """propagates the field u0 freely to distance dz

    
    """

    return _bpm_3d_free(size = size,
                        units = units,
                        dz = dz,
                        lam = lam,
                        n0 = n0,
                        use_fresnel_approx = use_fresnel_approx)

if __name__ == '__main__':
    pass
