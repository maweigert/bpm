"""the main method for beam propagation in media with coated spheres"""

import numpy as np

import volust
from volust.volgpu import OCLArray, OCLProgram
from volust.volgpu.oclfft import ocl_fft, ocl_fft_plan
from volust.volgpu.oclalgos import OCLReductionKernel

from bpm.utils import StopWatch, absPath



def bpm_3d(size, units, lam = .5, u0 = None, dn = None,           
           return_scattering = False,
           use_fresnel_approx = False):
    """
    simulates the propagation of monochromativ wave of wavelength lam with initial conditions u0 along z in a media filled with dn

    size     -    the dimension of the image to be calulcated  in pixels (Nx,Ny,Nz)
    units    -    the unit lengths of each dimensions in microns
    lam      -    the wavelength
    u0       -    the initial field distribution, if u0 = None an incident  plane wave is assumed
    dn       -    the refractive index of the medium (can be complex)

    """
    clock = StopWatch()

    clock.tic("setup")
    Nx, Ny, Nz = size
    dx, dy, dz = units

    

    #setting up the propagator
    k0 = 2.*np.pi/lam

    kxs = np.arange(-Nx/2.,Nx/2.)/Nx
    kys = np.arange(-Ny/2.,Ny/2.)/Ny

    KY, KX = np.meshgrid(kxs,kys, indexing= "ij")

    H0 = np.sqrt(0.j+(1./lam)**2-KX**2/dx**2-KY**2/dy**2)

    if use_fresnel_approx:
        H0  = 1./lam*(0.j+1.-.5*lam**2*(KX**2/dx**2+KY**2/dy**2))

        
    outsideInds = np.isnan(H0)
    H = np.exp(2.j*np.pi*dz*H0)
    H[outsideInds] = 0.
    H0[outsideInds] = 0.

    H = np.fft.fftshift(H).astype(np.complex64)


    if u0 is None:
        u0 = np.ones((Ny,Nx),np.complex64)

    """
    setting up the gpu buffers and kernels
    """

    program = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))
    # program = OCLProgram(src_str = kernel_str)

    plan = ocl_fft_plan((Ny,Nx))
    plane_g = OCLArray.from_array(u0.astype(np.complex64))

    h_g = OCLArray.from_array(H.astype(np.complex64))
    u_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.complex64)

    if dn is not None:
        if isinstance(dn,OCLArray):
            dn_g = dn
        else:
            if dn.dtype.type in (np.complex64,np.complex128):
                dn_g = OCLArray.from_array(dn.astype(np.complex64))
            else:
                dn_g = OCLArray.from_array(dn.astype(np.float32))
    else:
        dn_g = OCLArray.zeros((Nz,Ny,Nx),dtype=np.float32)

    isComplexDn = dn_g.dtype.type in (np.complex64,np.complex128)

    print isComplexDn
        
    if return_scattering:
        scatter_weights = np.fft.fftshift(np.real(H0)*
                                          np.sqrt(KX**2/dx**2+KY**2/dy**2))
        scatter_weights *= scatter_weights
        scatter_weights_g = OCLArray.from_array(scatter_weights.astype(np.float32))
        scatter_cross_sec_g = OCLArray.zeros(Nz,"float32")
        plain_wave_dct = Nx*Ny*np.exp(2.j*np.pi*np.arange(Nz)*dz/lam).astype(np.complex64)

        reduce_kernel = OCLReductionKernel(
        np.float32, neutral="0",
            reduce_expr="a+b",
            map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
            arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")

    u_g[0,...] = plane_g

 
    clock.toc("setup")
    clock.tic("run")

    for i in range(Nz-1):
        ocl_fft(plane_g,inplace = True, plan  = plan)

        program.run_kernel("mult",(Nx*Ny,),None,
                           plane_g.data,h_g.data)

        if return_scattering:
            scatter_cross_sec_g[i+1] = reduce_kernel(plane_g,
                                                     scatter_weights_g,
                                                     plain_wave_dct[i+1])
        
        ocl_fft(plane_g,inplace = True, inverse = True,  plan  = plan)

        if isComplexDn:
            program.run_kernel("mult_dn_complex",(Nx*Ny,),None,
                           plane_g.data,dn_g.data,
                           np.float32(k0*dz),
                           np.int32((i+1)*Nx*Ny))
        else:
            program.run_kernel("mult_dn",(Nx*Ny,),None,
                           plane_g.data,dn_g.data,
                           np.float32(k0*dz),
                           np.int32((i+1)*Nx*Ny))
            
        u_g[i+1,...] = plane_g

 

    clock.toc("run")

    print clock
    if return_scattering:
        return u_g.get(), dn_g.get(), scatter_cross_sec_g.get()/Nx/Ny
    else:
        return u_g.get(), dn_g.get()


def bpm_3d_free(size, units, dz, lam = .5, u0 = None,
           use_fresnel_approx = False):
    """
    """
    clock = StopWatch()

    clock.tic("setup")
    Nx, Ny = size
    dx, dy = units

    #setting up the propagator
    k0 = 2.*np.pi/lam

    kxs = np.arange(-Nx/2.,Nx/2.)/Nx
    kys = np.arange(-Ny/2.,Ny/2.)/Ny

    KY, KX = np.meshgrid(kxs,kys, indexing= "ij")

    H0 = np.sqrt(0.j+(1./lam)**2-KX**2/dx**2-KY**2/dy**2)

    if use_fresnel_approx:
        H0  = 1./lam*(0.j+1.-.5*lam**2*(KX**2/dx**2+KY**2/dy**2))

        
    outsideInds = np.isnan(H0)
    H = np.exp(2.j*np.pi*dz*H0)
    H[outsideInds] = 0.
    H0[outsideInds] = 0.

    H = np.fft.fftshift(H).astype(np.complex64)


    if u0 is None:
        u0 = np.ones((Ny,Nx),np.complex64)

    """
    setting up the gpu buffers and kernels
    """

    program = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))
    # program = OCLProgram(src_str = kernel_str)

    plan = ocl_fft_plan((Ny,Nx))
    plane_g = OCLArray.from_array(u0.astype(np.complex64))

    h_g = OCLArray.from_array(H.astype(np.complex64))

 
    clock.toc("setup")
    clock.tic("run")

    ocl_fft(plane_g,inplace = True, plan  = plan)

    program.run_kernel("mult",(Nx*Ny,),None,
                           plane_g.data,h_g.data)

    ocl_fft(plane_g,inplace = True, inverse = True,  plan  = plan)

    clock.toc("run")

    return plane_g.get()
    
if __name__ == '__main__':

    Nx, Nz = 128,128
    dx, dz = .05, 0.05

    lam = .5

    import imgtools
    Z,Y,X = imgtools.ZYX(dshape=(Nz,Nx,Nx))
    R = np.sqrt(X**2+Y**2+Z**2)
    dn = np.zeros((Nz,Nx,Nx))

    dn += .1* (R<.2)

    dn = dn*(0+1.j)
    
    u, dn, p = bpm_3d((Nx,Nx,Nz),(dx,dx,dz),
                      dn = dn, lam = lam,
                      return_scattering = True)

