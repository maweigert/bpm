"""the main method for beam propagation in refractive media"""

import numpy as np
from bpm.utils import StopWatch



#this is the main method to calculate everything

import pycuda.driver as cuda
import pycuda.autoinit as autoinit
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel


import reikna.cluda as cluda
from reikna.fft import FFT


ctx = autoinit.context

if __name__ == '__main__':


    api = cluda.cuda_api()
    thr = api.Thread.create()

    size = (256,256,256)
    units = (.1,)*3
    lam = .5
    u0 = None
    n0 = 1.
    dn = np.zeros(size[::-1],np.complex64)


    clock = StopWatch()

    clock.tic("setup")

    Nx, Ny, Nz = size
    dx, dy, dz = units

    dn[Nz/2:,...] = 0.1

    #setting up the propagator
    k0 = 2.*np.pi/lam

    kxs = 2.*np.pi*np.fft.fftfreq(Nx,dx)
    kys = 2.*np.pi*np.fft.fftfreq(Ny,dy)

    KY, KX = np.meshgrid(kxs,kys, indexing= "ij")

    #H0 = np.sqrt(0.j+n0**2*k0**2-KX**2-KY**2)
    H0 = np.sqrt(n0**2*k0**2-KX**2-KY**2)

    outsideInds = np.isnan(H0)

    H = np.exp(-1.j*dz*H0)

    H[outsideInds] = 0.
    H0[outsideInds] = 0.

    if u0 is None:
        u0 = np.ones((Ny,Nx),np.complex64)
    

    # setting up the gpu buffers and kernels
    dn_g = thr.to_device(dn.astype(np.float32))
    H_g = thr.to_device(H.astype(np.complex64))
    u_g = thr.array(size[::-1],np.complex64)
    plane_g = thr.to_device(u0.astype(np.complex64))
    fftobj = FFT(u0).compile(thr)


    mod = SourceModule("""
    #include <pycuda-complex.hpp>
    __global__ void mult_real(pycuda::complex<float> *data, float *dn, float kdz, int offset)
    {
        int i = threadIdx.x + threadIdx.x*blockDim.x;
        float dnval = dn[i+offset];
        pycuda::complex<float> tmp(cos(kdz*dnval),sin(kdz*dnval));
        data[i] *= tmp;

    }
    __global__ void mult_complex(pycuda::complex<float> *data,
    pycuda::complex<float> *b)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        data[i] *= b[i];
    }

    """)
    func_mult_real = mod.get_function("mult_real")
    func_mult_comp = mod.get_function("mult_complex")

    clock.toc("setup")


    for z in xrange(Nz):
        fftobj(plane_g,plane_g)
        func_mult_comp(plane_g,H_g,
                       grid = (Nx*Ny/256,1,1),
                       block=(256,1,1))

        fftobj(plane_g,plane_g, inverse = True)
        func_mult_real(plane_g, dn_g, np.float32(k0*dz), np.int32(z*Nx*Nx),
                       grid = (Nx*Ny/256,1,1),
                       block=(256,1,1))

        u_g[z] = plane_g

    u = u_g.get()




    clock.tic("run")


    for i in range(Nz-1):
        pass

    clock.toc("run")

    print clock

