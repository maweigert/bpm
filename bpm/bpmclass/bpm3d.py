"""
the main class for gpu accelerated bpm propagation

mweigert@mpi-cbg.de

"""

import numpy as np


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


class _Bpm3d_Base(object):
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

        #this part is implementation dependend...

        self.set_dn(dn)

        self._setup_impl()

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
        self.size2d = size[:2]
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
        #self._H0 = np.sqrt(0.j+n0**2*self.k0**2-self._KX**2-self._KY**2)

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




def Bpm3d(size, units, lam = .5, n0 = 1.,
            dn = None,
            use_fresnel_approx = False,
          n_volumes = 1,
          backend = "opencl"):

    """factory function for Bpm3d objects"""

    if backend=="opencl":
        from _bpm3d_ocl import _Bpm3d_OCL

        return _Bpm3d_OCL(size  = size,
                          units = units,
                          lam = lam,
                          n0 = n0,
                          dn = dn,
                          use_fresnel_approx = use_fresnel_approx,
                        n_volumes = n_volumes)
    else:
        raise NotImplementedError

Bpm3d.__doc__ = _Bpm3d_Base.__doc__


if __name__ == '__main__':

    from time import time

    t = time()
    m = Bpm3d((256,256,1000),(.1,)*3)
    print time()-t


    m._propagate_single()
