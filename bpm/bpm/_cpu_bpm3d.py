"""
the main class for gpu accelerated bpm propagation

mweigert@mpi-cbg.de

"""

import numpy as np
from bpm import psf_focus_u0


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



    def set_dn(self, dn = None):
        """
        :param dn:  the refractive index distribution as a float32/complex64 numpy array
        :return:
        """

        self.dn = dn


    def get_first_svd(self, x):
        u,s,v = linalg.svd(x)
        return np.outer(u[:,0],v[0,:])*s[0]


    def propagate(self, u0 = None, mode = "n0", **kwargs):
        """
        mode = "n0", "nmean", "svd"
        """
        #plane wave if none
        if u0 is None:
            u0 = np.ones(self.size[:2],np.complex64)

        Nx, Ny, Nz = self.size
        dx, dy, dz = self.units

        u = np.empty((Nz,Ny,Nx),dtype=np.complex64)
        u[0] = u0


        for i in range(Nz-1):
            u0_f = np.fft.fftn(u0)

            if mode == "n0":
                nmean = self.n0
            elif mode =="nmean":
                nmean = self.n0+np.mean(self.dn[i])
            elif mode =="svd":
                nmean = self.n0+np.mean(self.dn[i])
            else:
                raise ValueError()


            print nmean
            H0 = np.sqrt(nmean**2*self.k0**2-self._KX**2-self._KY**2)
            outsideInds = np.isnan(H0)

            H = np.exp(-1.j*dz*H0)
            H[outsideInds] = 0.

            u0_f *= H
            u0 = np.fft.ifftn(u0_f)

            if not self.dn is None:
                dn_slice = self.dn[i]+self.n0-nmean

                u0 *= np.exp(-1.j*dz*self.k0*dn_slice)


            u[i+1,...] = u0

        return u



    def __repr__(self):
        return "Bpm3d class with size %s and units %s"%(self.size,self.units)

if __name__ == '__main__':

    from time import time

    t = time()

    N = 256

    x = np.linspace(-1.,1.,N)
    Z,Y,X = np.meshgrid(x,x,x,indexing="ij")
    R = np.sqrt(X**2+Y**2+(Z+.4)**2)
    dn = .7*(R<.4)

    m = Bpm3d((N,)*3,(.1,)*3)

    m.set_dn(dn)

    u0 = psf_focus_u0(m.size[:2],m.units[:2],lam = 0.5, zfoc=0, NAs = .8)

    u1 = m.propagate(u0 = u0, mode = "n0")

    u2 = m.propagate(u0 = u0, mode = "nmean")

    us = [u1,u2]

    import pylab
    pylab.figure(1)
    pylab.clf()
    for i,u in enumerate(us):
        pylab.subplot(2,len(us),i+1)
        pylab.imshow(abs(u[...,N/2]),cmap = "hot")
        pylab.axis("off")

    pylab.subplot(2,len(us),len(us)+1)
    for i,u in enumerate(us):
        pylab.plot(abs(u[:,N/2,N/2]),label = str(i))
    pylab.legend()
    pylab.show()
    print time()-t
