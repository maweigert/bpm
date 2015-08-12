"""the main method for beam propagation in refractive media"""

import numpy as np

import volust
from volust.volgpu import OCLArray, OCLImage, OCLProgram, get_device
from volust.volgpu.oclfft import ocl_fft, ocl_fft_plan
from volust.volgpu.oclalgos import OCLReductionKernel

from bpm.utils import StopWatch, absPath

from scipy.ndimage.interpolation import zoom

#this is the main method to calculate everything

def bpm_3d(size,
           units,
           lam = .5,
           u0 = None, dn = None,
           subsample = 1,
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
    u0       -    the initial field distribution, if u0 = None an incident  plane wave is assumed
    dn       -    the refractive index of the medium (can be complex)
    n0       -    refractive index of surrounding medium
    """

    if n_volumes ==1:
        return _bpm_3d(size, units,
                       lam = lam,
                       u0 = u0, dn = dn,
                       subsample = subsample,
                       n0 = n0,
                       return_scattering = return_scattering,
                       return_g = return_g,
                       use_fresnel_approx = use_fresnel_approx)
    else:
        return _bpm_3d_split(size, units,
                             lam = lam,
                             u0 = u0, dn = dn,
                             n_volumes = n_volumes,
                             subsample = subsample,
                             n0 = n0,                            
                             return_scattering = return_scattering,
                             return_g = return_g,
                             use_fresnel_approx = use_fresnel_approx)


def _bpm_3d(size,
            units,
            lam = .5,
            u0 = None, dn = None,
            subsample = 1,
            n0 = 1.,
            return_scattering = False,
            return_g = False,
            return_full_last = False,
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

    # subsampling
    Nx2, Ny2, Nz2 = (subsample*N for N in size)
    dx2, dy2, dz2 = (1.*d/subsample for d in units)

    #setting up the propagator
    k0 = 2.*np.pi/lam*n0

    kxs = 2.*np.pi*np.fft.fftfreq(Nx2,dx2)
    kys = 2.*np.pi*np.fft.fftfreq(Ny2,dy2)

    KY, KX = np.meshgrid(kxs,kys, indexing= "ij")

    H0 = np.sqrt(0.j+k0**2-KX**2-KY**2)
    H0 = np.sqrt(k0**2-KX**2-KY**2)

    if use_fresnel_approx:
        H0  = 0.j+k0-.5*(KX**2+KY**2)

    
    outsideInds = np.isnan(H0)

    H = np.exp(1.j*dz2*H0)

    H[outsideInds] = 0.
    H0[outsideInds] = 0.

    if u0 is None:
        u0 = np.ones((Ny2,Nx2),np.complex64)
    else:
        if subsample >1:
            u0 = zoom(np.real(u0),subsample) + 1.j*zoom(np.imag(u0),subsample) 

    # setting up the gpu buffers and kernels

    program = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))

    plan = ocl_fft_plan((Ny2,Nx2))
    plane_g = OCLArray.from_array(u0.astype(np.complex64))

    h_g = OCLArray.from_array(H.astype(np.complex64))

    if dn is not None:
        if isinstance(dn,OCLImage):
            dn_g = dn
        else:
            if dn.dtype.type in (np.complex64,np.complex128):
                dn_g = OCLImage.from_array(dn.astype(np.complex64))
            else:
                dn_g = OCLImage.from_array(dn.astype(np.float32))
        
        isComplexDn = dn_g.dtype.type in (np.complex64,np.complex128)
    else:
        #dummy dn
        dn_g = OCLArray.empty((1,)*3,np.float16)

        
    if return_scattering:
        # = cos(theta)
        scatter_weights = np.real(H0)

        scatter_weights_g = OCLArray.from_array(scatter_weights.astype(np.float32))

        # = cos(theta)^2
        gfactor_weights = np.real(H0)**2
        
        gfactor_weights_g = OCLArray.from_array(gfactor_weights.astype(np.float32))


        scatter_cross_sec_g = OCLArray.zeros(Nz,"float32")
        gfactor_g = OCLArray.zeros(Nz,"float32")

        plain_wave_dct = Nx2*Ny2*np.exp(1.j*k0*np.arange(Nz)*dz).astype(np.complex64)

        reduce_kernel = OCLReductionKernel(
        np.float32, neutral="0",
            reduce_expr="a+b",
            map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
            arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")

        reduce_gfactor_kernel = OCLReductionKernel(
        np.float32, neutral="0",
            reduce_expr="a+b",
            map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
            arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")


    u_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.complex64)

    program.run_kernel("copy_subsampled_buffer",(Nx,Ny),None,
                           u_g.data,plane_g.data,
                           np.int32(subsample),
                           np.int32(0))

    
    clock.toc("setup")
    
    clock.tic("run")

    for i in range(Nz-1):
        for substep in range(subsample):
            ocl_fft(plane_g,inplace = True, plan  = plan)

            program.run_kernel("mult",(Nx2*Ny2,),None,
                               plane_g.data,h_g.data)
            
            if return_scattering and substep == (subsample-1):
                scatter_cross_sec_g[i+1] = reduce_kernel(plane_g,
                                                     scatter_weights_g,
                                                     plain_wave_dct[i+1])
                gfactor_g[i+1] = reduce_gfactor_kernel(plane_g,
                                                     gfactor_weights_g,
                                                     plain_wave_dct[i+1])

            ocl_fft(plane_g,inplace = True, inverse = True,  plan  = plan)

            if dn is not None:
                if isComplexDn:
                    program.run_kernel("mult_dn_complex_image",(Nx2,Ny2),None,
                                   plane_g.data,dn_g,
                                   np.float32(k0*dz2),
                                   np.float32(n0),                               
                                   np.int32(subsample*(i+1.)+substep),
                                   np.int32(subsample))
                else:
                    program.run_kernel("mult_dn_image",(Nx2,Ny2),None,
                                   plane_g.data,dn_g,
                                   np.float32(k0*dz2),
                                   np.float32(n0),                                                                  
                                   np.int32(subsample*(i+1.)+substep),
                                   np.int32(subsample))

                
        program.run_kernel("copy_subsampled_buffer",(Nx,Ny),None,
                           u_g.data,plane_g.data,
                           np.int32(subsample),
                           np.int32((i+1)*Nx*Ny))
                
        
    clock.toc("run")

    print clock
    result = (u_g.get(), dn_g.get(),)
    
    if return_scattering:
        prefac = 1./Nx2/Ny2*dx2*dy2/4./np.pi
        result += (prefac*scatter_cross_sec_g.get(),)
        
    if return_g:
        prefac = 1./Nx/Ny*dx*dy/4./np.pi        
        result += (prefac*gfactor_g.get(),)
        
    if return_full_last:
        result += (plane_g.get(),)

    return result


def bpm_3d_old(size, units, lam = .5, u0 = None, dn = None,
           return_scattering = False,
           return_g = False,
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

    kxs = 2.*np.pi*np.fft.fftfreq(Nx,dx)
    kys = 2.*np.pi*np.fft.fftfreq(Ny,dy)

    KY, KX = np.meshgrid(kxs,kys, indexing= "ij")

    H0 = np.sqrt(0.j+k0**2-KX**2-KY**2)

    if use_fresnel_approx:
        H0  = 0.j+k0-.5*(KX**2+KY**2)

    
    outsideInds = np.isnan(H0)
    H = np.exp(1.j*dz*H0)
    H[outsideInds] = 0.
    H0[outsideInds] = 0.

    # kxs = np.arange(-Nx/2.,Nx/2.)/Nx
    # kys = np.arange(-Ny/2.,Ny/2.)/Ny

    # KY, KX = np.meshgrid(kxs,kys, indexing= "ij")

    # H0 = np.sqrt(0.j+(1./lam)**2-KX**2/dx**2-KY**2/dy**2)

    # if use_fresnel_approx:
    #     H0  = 1./lam*(0.j+1.-.5*lam**2*(KX**2/dx**2+KY**2/dy**2))


    # outsideInds = np.isnan(H0)
    # H = np.exp(2.j*np.pi*dz*H0)
    # H[outsideInds] = 0.
    # H0[outsideInds] = 0.
    
    # H = np.fft.fftshift(H).astype(np.complex64)

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

        
    if return_scattering:
        # scatter_weights = np.fft.fftshift(np.real(H0)*
        #                                   np.sqrt(KX**2/dx**2+KY**2/dy**2))
        
        # scatter_weights *= scatter_weights


        scatter_weights = np.real(H0)

        # scatter_weights = np.real(H0)*np.sqrt(KX**2/k0**2+KY**2/k0**2)
        
        scatter_weights_g = OCLArray.from_array(scatter_weights.astype(np.float32))

        #which is the cosine
        gfactor_weights = np.real(H0)**2
        
        gfactor_weights_g = OCLArray.from_array(gfactor_weights.astype(np.float32))


        scatter_cross_sec_g = OCLArray.zeros(Nz,"float32")
        gfactor_g = OCLArray.zeros(Nz,"float32")

        plain_wave_dct = Nx*Ny*np.exp(1.j*k0*np.arange(Nz)*dz).astype(np.complex64)

        # reduce_kernel = OCLReductionKernel(
        # np.float32, neutral="0",
        #     reduce_expr="a+b",
        #     map_expr="weights[i]*cfloat_abs(field[i])*cfloat_abs(field[i])",
        #     arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")

        reduce_kernel = OCLReductionKernel(
        np.float32, neutral="0",
            reduce_expr="a+b",
            map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
            arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")

        reduce_gfactor_kernel = OCLReductionKernel(
        np.float32, neutral="0",
            reduce_expr="a+b",
            map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
            arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")

    u_g[0,...] = plane_g
    clock.toc("setup")
    
    clock.tic("run")

    n0 = 1.
    for i in range(Nz-1):
        ocl_fft(plane_g,inplace = True, plan  = plan)

        program.run_kernel("mult",(Nx*Ny,),None,
                           plane_g.data,h_g.data)

        if return_scattering:
            scatter_cross_sec_g[i+1] = reduce_kernel(plane_g,
                                                     scatter_weights_g,
                                                     plain_wave_dct[i+1])
            gfactor_g[i+1] = reduce_gfactor_kernel(plane_g,
                                                     gfactor_weights_g,
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
        prefac = 1./Nx/Ny*dx*dy/4./np.pi
        if return_g:
            return u_g.get(), dn_g.get(), prefac*scatter_cross_sec_g.get(),
            prefac*gfactor_g.get()
        else:
            return u_g.get(), dn_g.get(), prefac*scatter_cross_sec_g.get()
    else:
        return u_g.get(), dn_g.get() 

def _bpm_3d_split(size, units, lam = .5, u0 = None, dn = None,
                 n_volumes = 1,
                  n0 = 1.,
                  subsample=1,
                 return_scattering = False,
                 return_g = False,
                 use_fresnel_approx = False):
    """
    same as bpm_3d but splits z into n_volumes pieces (e.g. if memory of GPU is not enough)
    """
    
    Nx, Ny, Nz = size

    Nz2 = Nz/n_volumes+1

    if u0 is None:
        u0 = np.ones((subsample*Ny,subsample*Nx),np.complex64)

    if dn is None:
        dn = np.zeros((Nz,Ny,Nx),np.float32)
    
    u = np.empty((Nz,Ny,Nx),np.complex64)

    p = np.empty(Nz,np.float32)
    
    u_part = np.empty((Nz2,Ny,Nx),np.complex64)

    for i in range(n_volumes):
        i1,i2 = i*Nz2, np.clip((i+1)*Nz2,0,Nz)
        if i<n_volumes-1:
            res_part = _bpm_3d((Nx,Ny,i2-i1+1),
                               units = units,
                               lam = lam,
                               u0 = u0,
                               dn = dn[i1:i2+1,:,:],
                               n0 = n0,
                               subsample = subsample,
                               return_full_last = True,
                               return_scattering = return_scattering)
            if return_scattering:
                u_part, _, p_part, u0 = res_part
                p[i1:i2] = p_part[1:]
            else:
                u_part, _ , u0 = res_part

            u[i1:i2,...] = u_part[1:,...]
            
        else:
            res_part = _bpm_3d((Nx,Ny,i2-i1),
                               units = units,
                               lam = lam,
                               u0 = u0,
                               dn = dn[i1:i2,:,:],
                               n0 = n0,                               
                               subsample = subsample,
                               return_full_last = True,
                               return_scattering = return_scattering)

            if return_scattering:
                u_part, _, p_part, u0 = res_part
                p[i1:i2] = p_part
                p[i1] = p[i1-1]
            else:
                u_part, _, u0 = res_part

            u[i1:i2,...] = u_part
            
    if return_scattering:
        return u, None,p
    else:
        return u, None
    
def bpm_3d_free(size, units, dz, lam = .5, u0 = None,
                n0 = 1., 
                use_fresnel_approx = False):
    """propagates the field u0 to distance dz
    """
    clock = StopWatch()

    clock.tic("setup")
    Nx, Ny = size
    dx, dy = units

    #setting up the propagator
    k0 = 2.*np.pi/lam*n0

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
    

def test_3d():                        
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


def test_speed():
    from time import time
    
    for N in [64,128,256,512,1024][2:]:
        shape = (N,N,64)
        bpm_3d(shape,units= (.1,)*3)

        Niter = 3
        t = time()
        for i in range(Niter):
            bpm_3d(shape,units= (.1,)*3)
        
        print "time to bpm through %s = %.3f ms"%(shape,1000.*(time()-t)/Niter)

def test_plane(n_x_comp = 0, n0 = 1.):
    """ propagates a plane wave freely
    n_x_comp is the tilt in x
    """
    Nx, Nz = 128,128
    dx, dz = .05, 0.05

    lam = .5

    units = (dx,dx,dz)
    
    x = dx*np.arange(Nx)
    y = dx*np.arange(Nx)
    z = dz*np.arange(Nz)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")

    
    k_x = 1.*n_x_comp/(dx*(Nx-1.))

    
    k_z = np.sqrt(1.*n0**2/lam**2-k_x**2)

    print np.sqrt(k_x**2+k_z**2)
    
    u_plane = np.exp(2.j*np.pi*(k_z*Z+k_x*X))

    u = 0

    u, dn = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
                   n0 = n0,
                   subsample = 2,
                   u0 = u_plane[0,...])

    # u, dn = bpm_3d_old((Nx,Nx,Nz),units= units, lam = lam,
    #                u0 = u_plane[0,...])
    
    print np.mean(np.abs(u_plane-u)**2)
    return u, u_plane

def test_slit():
    Nx, Nz = 128,128
    dx, dz = .05, 0.05

    lam = 0.5

    units = (dx,dx,dz)

    
    
    x = np.linspace(-1,1,Nx)
    y = np.linspace(-1,1,Nx)
    Y,X = np.meshgrid(y,x,indexing="ij")

    R = np.hypot(Y,X)

    u0 = 1.*(R<.5) 

    
    u, dn, p = bpm_3d((Nx,Nx,Nz),units= units,
                      lam = lam,
                      u0 = u0,
                      dn = np.zeros((Nz,Nx,Nx)),
                      subsample = 1,
                      return_scattering = True )

    return u, dn, p


def test_sphere():
    Nx, Nz = 128,128
    dx, dz = .05, 0.05

    lam = .5

    units = (dx,dx,dz)
    
    x = Nx/2*dx*np.linspace(-1,1,Nx)
    y = Nx/2*dx*np.linspace(-1,1,Nx)
    
    x = dx*np.arange(-Nx/2,Nx/2)
    y = dx*np.arange(-Nx/2,Nx/2)
    z = dz*np.arange(0,Nz)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")
    R = np.sqrt(X**2+Y**2+(Z-3.)**2)
    dn = .05*(R<1.)
    
    u, dn, p = bpm_3d((Nx,Nx,Nz),units= units,
                      lam = lam,
                      dn = dn,
                      subsample = 1,
                      n_volumes = 1,
                      return_scattering = True )

    
    print np.sum(np.abs(u[1:,...]))
    return u, dn,p

def test_compare():
    Nx, Nz = 128,256
    dx, dz = .05, 0.05

    lam = .5

    units = (dx,dx,dz)

    
    
    x = Nx/2*dx*np.linspace(-1,1,Nx)
    y = Nx/2*dx*np.linspace(-1,1,Nx)
    
    x = dx*np.arange(-Nx/2,Nx/2)
    y = dx*np.arange(-Nx/2,Nx/2)
    z = dz*np.arange(0,Nz)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")
    R = np.sqrt(X**2+Y**2+(Z-3.)**2)
    dn = .05*(R<1.)
    
    u1, dn1, p1 = bpm_3d((Nx,Nx,Nz),units= units,
                      lam = lam,
                      dn = dn,
                      subsample = 1,
                      n_volumes = 1,
                      return_scattering = True )

    u2, dn2, p2 = bpm_3d_old((Nx,Nx,Nz),units= units,
                      lam = lam,
                      dn = dn,
                      return_scattering = True )
    
    return u1, u2


if __name__ == '__main__':
    # test_speed()

    u, dn, p = test_sphere()

    # u1, u2 = test_compare()

    
    # u, u0 = test_plane(n_x_comp = 1 , n0 = 1.)

    # u, dn, p = test_slit()


    
    # Nx, Nz = 256,128
    # dx, dz = .05, 0.05

    # lam = .5

    # units = (dx,dx,dz)
    # rad = 2.

    # x = dx*np.arange(-Nx/2,Nx/2)
    # x = dx*np.arange(-Nx/2,Nx/2)
    # z = dz*np.arange(Nz)
    # Z,Y,X = np.meshgrid(z,x,x,indexing="ij")
    # R = np.sqrt(X**2+Y**2)
    # dn = ((R<2.)*(Z<.1))*100.j
    
    # u, dn, p = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
    #                   dn = dn,
    #                   return_scattering = True )

    # u0 = u[1,...]
    # u = bpm_3d_split((Nx,Nx,Nz),units= units, NZsplit=4,lam = lam)

    
