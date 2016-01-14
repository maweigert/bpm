"""


mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan
from gputools import OCLReductionKernel

from bpm.utils import StopWatch, absPath

class Bpm3d_Base(object):
    """
    the main class for bpm propagation
    """
    def __init__(self, size, units, lam = .5, n0 = 1., bpm_kwargs = {}):
        """

        :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
        :param units: the physical units of each voxel in microns (dx,dy,dz)
        :param lam: the wavelength of light in microns
        :param n0:  the refractive index of the surrounding media
        :param bpm_kwargs: keywords used in propagate
        example:

        model = Bpm3d(size = (128,128,128),
                      units = (0.1,0.1,0.1),
                      lam = 0.5,
                      n0 = 1.33)
        """
        self.setup(size = size, units = units, lam = lam, n0 = n0)

    def _setup(self, size, units, lam = .5, n0 = 1.,
              use_fresnel_approx = False):
        """
            sets up the internal variables e.g. propagators etc...

            :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
            :param units: the phyiscal units of each voxel in microns (dx,dy,dz)
            :param lam: the wavelength of light in microns
            :param n0:  the refractive index of the surrounding media
            :param use_fresnel_approx:  if True, uses fresnel approximation for propagator


        """
        self.size = size
        self.units = units
        self.n0 = n0

        Nx, Ny, Nz = size
        dx, dy, dz = units


        #setting up the propagator
        self.k0 = 2.*np.pi/lam

        kxs = 2.*np.pi*np.fft.fftfreq(Nx,dx)
        kys = 2.*np.pi*np.fft.fftfreq(Ny,dy)

        self._KY, self._KX = np.meshgrid(kxs,kys, indexing= "ij")

        self._H0 = np.sqrt(n0**2*self.k0**2-self._KX**2-self._KY**2)

        if use_fresnel_approx:
            self._H0  = 0.j+n0**2*self.k0-.5*(self._KX**2+self._KY**2)


        outsideInds = np.isnan(self._H0)

        self._H = np.exp(-1.j*dz*self._H0)

        self._H[outsideInds] = 0.
        self._H0[outsideInds] = 0.

        #scattering
        self._cos_theta = np.real(self._H0)/self.n0/self.k0

        self.scatter_weights = self._cos_theta
        self.gfactor_weights = cos_theta**2



    def set_dn(self, dn = None):
        assert self.size == dn.shape[::-1]
        self.dn = dn


    def propagate(self, u0 = None, **kwargs):
        """
        :param u0: initial complex field distribution, if None, plane wave is assumed
        :param kwargs:
        :return:
        """
        raise NotImplementedError()



class Bpm3d_CPU(Bpm3d_Base):
    def propagate(self, u0 = None, **kwargs):
        """
        :param u0: initial complex field distribution, if None, plane wave is assumed
        :param kwargs:
        :return:
        """



class Bpm3d_GPU(Bpm3d_Base):
    def setup(self, size, units, lam = .5, n0 = 1.,
              use_fresnel_approx = False):
        """
            sets up the internal variables e.g. propagators etc...

            :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
            :param units: the phyiscal units of each voxel in microns (dx,dy,dz)
            :param lam: the wavelength of light in microns
            :param n0:  the refractive index of the surrounding media
            :param use_fresnel_approx:  if True, uses fresnel approximation for propagator


        """
        Bpm3d_Base.setup(self,size, units, lam = lam, n0 = n0,
              use_fresnel_approx = use_fresnel_approx)

        #setting up the gpu buffers and kernels
        self.program = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))

        Nx, Ny  = self.size[:2]
        plan = fft_plan(())
        self._H_g = OCLArray.from_array(self._H.astype(np.complex64))


        self.scatter_weights_g = OCLArray.from_array(self.scatter_weights.astype(np.float32))
        self.gfactor_weights_g = OCLArray.from_array(self.gfactor_weights.astype(np.float32))

        self.scatter_cross_sec_g = OCLArray.zeros(Nz,"float32")
        self.gfactor_g = OCLArray.zeros(Nz,"float32")



        self.reduce_kernel = OCLReductionKernel(
        np.float32, neutral="0",
            reduce_expr="a+b",
            map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
            arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")





if __name__ == '__main__':


    m = Bpm3d((128,)*3,(.1,)*3)