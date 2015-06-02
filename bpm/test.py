"""the inverted method for beam propagation """

import numpy as np

import volust
from volust.volgpu import OCLArray, OCLProgram
from volust.volgpu.oclfft import ocl_fft, ocl_fft_plan
from volust.volgpu.oclalgos import OCLElementwiseKernel

from bpm.utils import StopWatch, absPath

from bpm.bpm_3d_spheres import bpm_3d_spheres
from bpm.bpm_3d_inverse  import bpm_3d_inverse
from bpm.bpm_3d import bpm_3d

from imgtools import convolve_sep3

from numpy import *

if __name__ == '__main__':
                           
    Nx, Nz = 256,64
    dx, dz = .02, 0.02

    lam = .5


    u0, dn0 = bpm_3d_spheres((Nx,Nx,Nz),units= (dx,dx,dz), lam = lam,
                              points = [[Nx*dx/2.,Nx*dx/2.,2.]],
                           dn_inner = .0, rad_inner=0, dn_outer=.1, rad_outer=2.)


    dn  = np.zeros_like(dn0)

    h = exp(-10*linspace(-1,1,21)**2)
    h*= 1./sum(h)

    for i in range(1):
        u, dn = bpm_3d((Nx,Nx,Nz),units= (dx,dx,dz), lam = lam,
                       dn = dn)

        u *= abs(u0)/(abs(u)+1.e-20)
        dn = bpm_3d_inverse(u,units = (dx,dx,dz),lam = lam)

        # dn = convolve_sep3(1.*real(dn),h,h,h)
        print np.amax(np.real(dn))
