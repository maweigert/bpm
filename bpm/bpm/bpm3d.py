"""
the main class for gpu accelerated bpm propagation

mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan
from gputools import OCLReductionKernel

from bpm.utils import StopWatch, absPath


def absPath(myPath):
    import sys
    import os
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


class Bpm3d(object):
    """
    the main class for gpu accelerated bpm propagation
    """

    _float_type = np.float32
    _complex_type = np.complex64

    def __init__(self, size, units, lam = .5, n0 = 1.,
                 dn = None,
                 use_fresnel_approx = False, n_volumes = 1):
        """

        :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
        :param units: the physical units of each voxel in microns (dx,dy,dz)
        :param dn: refractive index distribution (can be given later)
        :param lam: the wavelength of light in microns
        :param n0:  the refractive index of the surrounding media
        :param n_volumes: splits the domain into chunks if GPU memory is not
                        large enough

        example:

        model = Bpm3d(size = (128,128,128),
                      units = (0.1,0.1,0.1),
                      lam = 0.5,
                      n0 = 1.33)
        """

        self.n_volumes = n_volumes

        self._setup(size = size, units = units, lam = lam, n0 = n0,
                    use_fresnel_approx=False)
        self.set_dn(dn)






    def _setup(self, size, units, lam = .5, n0 = 1.,
              use_fresnel_approx = False):
        """
            sets up the internal variables e.g. propagators etc...

            :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
            :param units: the phyiscal units of each voxel in microns (dx,dy,dz)
            :param lam: the wavelength of light in microns
            :param n0:  the refractive index of the surrounding media,
            dn=None means free propagation
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

        #this is used for scattering calculations
        self._cos_theta = np.real(self._H0)/self.n0/self.k0

        self.scatter_weights = self._cos_theta
        self.gfactor_weights = self._cos_theta**2

        self.plain_wave_dct = Nx*Ny*np.exp(-1.j*self.k0*n0*np.arange(Nz)*dz).astype(np.complex64)

        # set up the gpu specific attributes
        self._setup_gpu()


    def _setup_gpu(self):
        """setting up the gpu buffers and kernels
        """

        self.bpm_program = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))

        Nx, Ny, Nz  = self.size

        self._plan = fft_plan((Ny,Nx))


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


    def set_dn(self, dn = None):
        """
        :param dn:  the refractive index distribution as a float32/complex64 numpy array
        :return:
        """

        if dn is None:
            self.dn = None
            return

        else:
            if self.size != dn.shape[::-1]:
                raise ValueError("size of dn  %s doesn't match internal size %s"%(dn.shape[::-1],self.size))

        if np.iscomplexobj(dn):
            self._is_complex_dn = True
            self.dn = dn.astype(self._complex_type, copy = False)
        else:
            self._is_complex_dn = False
            self.dn = dn.astype(self._float_type, copy = False)

        if self.n_volumes == 1:
           self.dn_g = OCLArray.from_array(dn)





    def _propagate_single(self, u0 = None, return_full = True, absorbing_width = 0, **kwargs):
        """
        :param u0: initial complex field distribution, if None, plane wave is assumed
        :param kwargs:
        :return:
        """


        #plane wave if none
        if u0 is None:
            u0 = np.ones(self.size2d,np.complex64)


        plane_g = OCLArray.from_array(u0.astype(np.complex64,copy = False))


        if return_full:
            u_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.complex64)
            u_g[0] = plane_g

        clock.toc("setup")

        clock.tic("run")


    # for i in range(Nz-1):
    #     fft(plane_g,inplace = True, plan  = plan)
    #
    #     program.run_kernel("mult",(Nx*Ny,),None,
    #                        plane_g.data,h_g.data)
    #
    #
    #
    #     if return_scattering:
    #         scatter_cross_sec_g[i+1] = reduce_kernel(plane_g,
    #                                                  scatter_weights_g,
    #                                                  plain_wave_dct[i+1])
    #         gfactor_g[i+1] = reduce_kernel(plane_g,
    #                                                  gfactor_weights_g,
    #                                                  plain_wave_dct[i+1])
    #
    #     fft(plane_g,inplace = True, inverse = True,  plan  = plan)
    #
    #     if dn is not None:
    #         if isComplexDn:
    #
    #             kernel_str = "mult_dn_complex"
    #         else:
    #
    #             kernel_str = "mult_dn"
    #
    #
    #         program.run_kernel(kernel_str,(Nx,Ny,),None,
    #                                plane_g.data,dn_g.data,
    #                                np.float32(k0*dz),
    #                                np.int32(Nx*Ny*(i+1)),
    #                            np.int32(absorbing_width))
    #
    #
    #
    #
    #     if return_full:
    #         u_g[i+1] = plane_g
    #
    # clock.toc("run")
    #
    # print clock
    #
    # if return_full:
    #     u = u_g.get()
    # else:
    #     u = plane_g.get()
    #
    # if return_scattering:
    #     # normalizing prefactor dkx = dx/Nx
    #     # prefac = 1./Nx/Ny*dx*dy/4./np.pi/n0
    #     prefac = 1./Nx/Ny*dx*dy
    #     p = prefac*scatter_cross_sec_g.get()
    #
    #
    # if return_g:
    #     prefac = 1./Nx/Ny*dx*dy
    #     g = prefac*gfactor_g.get()/p
    #
    # if return_scattering:
    #     if return_g:
    #         return u,  p, g
    #     else:
    #         return u,  p
    # else:
    #     return u

        raise NotImplementedError()


    def __repr__(self):
        return "Bpm3d class with size %s and units %s"%(self.size,self.units)

if __name__ == '__main__':

    from time import time

    t = time()
    m = Bpm3d((256,)*3,(.1,)*3)
    print time()-t


    m._propagate_single()
