"""the inverted method for beam propagation """

import numpy as np

import volust
from volust.volgpu import OCLArray, OCLProgram
from volust.volgpu.oclfft import ocl_fft, ocl_fft_plan
from volust.volgpu.oclalgos import OCLElementwiseKernel

from bpm.utils import StopWatch, absPath

from bpm.bpm_3d_spheres import bpm_3d_spheres


def bpm_3d_inverse(u,units, lam = .5,
           use_fresnel_approx = False):
    """
    size     -    the dimension of the image to be calulcated  in pixels (Nx,Ny,Nz)
    units    -    the unit lengths of each dimensions in microns
    lam      -    the wavelength
    u       -    the complex field distribution

    returns 
    dn       -    the refractive index of the medium (can be complex)

    """
    clock = StopWatch()

    clock.tic("setup")
    Nz, Ny, Nx = u.shape
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


    """
    setting up the gpu buffers and kernels
    """

    program = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))
    # program = OCLProgram(src_str = kernel_str)

    kernel_divide = OCLElementwiseKernel(
        "cfloat_t *a_g, cfloat_t *b_g,float kz, cfloat_t *res_g",
        "res_g[i] = (cfloat_t)(i,0.)",
    "divide")
    
    plan = ocl_fft_plan((Ny,Nx))
    plane_g = OCLArray.empty((Ny,Nx),np.complex64)
    plane0_g = OCLArray.empty((Ny,Nx),np.complex64)

    h_g = OCLArray.from_array(H.astype(np.complex64))
    u_g = OCLArray.from_array(u.astype(np.complex64))

    dn_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.complex64)

 
    clock.toc("setup")
    clock.tic("run")

    for i in range(Nz-1):
        program.run_kernel("copy_complex",(Nx*Ny,),None,
                           u_g.data,plane_g.data,np.int32(i*Nx*Ny))

        #calculate the propagated plane
        ocl_fft(plane_g,inplace = True, plan  = plan)

        program.run_kernel("mult",(Nx*Ny,),None,
                           plane_g.data,h_g.data)

        
        ocl_fft(plane_g,inplace = True, inverse = True,  plan  = plan)

        dn_g[i+1,...] = plane_g
        
        # program.run_kernel("copy_complex",(Nx*Ny,),None,
        #                    u_g.data,plane0_g.data,np.int32((i+1)*Nx*Ny))

        
        # program.run_kernel("divide_dn_complex",(Nx*Ny,),None,
        #                    plane0_g.data,plane_g.data,dn_g.data,
        #                    np.float32(k0*dz),
        #                    np.int32((i+1)*Nx*Ny))


 


    clock.toc("run")

    print clock
    return dn_g.get()


if __name__ == '__main__':
                           
    Nx, Nz = 128,256
    dx, dz = .05, 0.05

    lam = .5


    u, dn = bpm_3d_spheres((Nx,Nx,Nz),units= (dx,dx,dz), lam = lam,
                              points = [[Nx*dx/2.,Nx*dx/2.,2.5],
#                                    [Nx*dx/2.,Nx*dx/2.,7.5]
                                    ],
                           dn_inner = .0, rad_inner=0, dn_outer=.2, rad_outer=2.5)


    dn2 = bpm_3d_inverse(u,units = (dx,dx,dz),lam = lam)
    

    
