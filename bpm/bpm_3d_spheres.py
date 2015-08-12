"""the main method for beam propagation in media with coated spheres"""

import numpy as np

from gputools import OCLArray, OCLProgram, get_device

from bpm.utils import absPath
from bpm.bpm_3d import bpm_3d


def create_dn_buffer(size, units,points,
                     dn_inner = .0, rad_inner = 0,
                     dn_outer = .1, rad_outer = .4):

    Nx, Ny, Nz = size
    dx, dy, dz = units

    program = OCLProgram(absPath("kernels/bpm_3d_spheres.cl"))


    dn_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.float32)

    # sort by z
    ps = np.array(points)
    ps = ps[np.argsort(ps[:,2]),:]

    Np = ps.shape[0]

    pointsBuf = OCLArray.from_array(ps.flatten().astype(np.float32))

    program.run_kernel("fill_dn",(Nx,Ny,Nz),None,dn_g.data,
                       pointsBuf.data,np.int32(Np),
                       np.float32(dx),np.float32(dy),np.float32(dz),
                       np.float32(dn_inner),np.float32(rad_inner),
                       np.float32(dn_outer),np.float32(rad_outer))


    return dn_g



def bpm_3d_spheres(size, units, lam = .5, u0 = None,
                   points = None,
                   dn_inner = 0.0, rad_inner = 0.,                   
                   dn_outer = 0.05, rad_outer = 2.5,
                   return_scattering = False,
                   use_fresnel_approx = False
):
    """
    simulates the propagation of monochromativ wave of wavelength lam with initial conditions u0 along z in a media filled with coated spheres at positions points and inner radius r1 and dn1 and outer radius r2/ dn2

    size     -    the dimension of the image to be calulcated  in pixels (Nx,Ny,Nz)
    units    -    the unit lengths of each dimensions in microns
    lam      -    the wavelength
    u0       -    the initial field distribution, if u0 = None an incident  plane wave is assumed
    points   -    a list of coordinates [[x1,y1,z1],...] in real unit interval
                  [0...dx*nx] where spheres are placed.

    """
    dn_g = create_dn_buffer(size,units,points = points,
                   dn_inner = dn_inner, rad_inner = rad_inner,                   
                   dn_outer = dn_outer, rad_outer = rad_outer)

    return bpm_3d(size = size,units = units,lam = lam,
                  u0 = u0,dn = dn_g,
                  return_scattering= return_scattering,
                  use_fresnel_approx = use_fresnel_approx)


def bpm_3d_spheres_split(size, units, NZsplit = 1,lam = .5, u0 = None,
                   points = None,
                   dn_inner = 0.04, rad_inner = 3.,                   
                   dn_outer = 0.02, rad_outer = 4.,
                   return_scattering = True,
                   use_fresnel_approx = False
):    
    """
    same as bpm_3d_spheres but splits z into Nz pieces (e.g. if memory of GPU is not enough)
    """
    
    Nx, Ny, Nz = size

    Nz2 = Nz/NZsplit+1

    if u0 is None:
        u0 = np.ones((Ny,Nx),np.complex64)

    u = np.empty((Nz,Ny,Nx),np.complex64)
    u_part = np.empty((Nz2,Ny,Nx),np.complex64)
   
    dn = np.empty((Nz,Ny,Nx),np.float32)
    dn_part = np.empty((Nz2,Ny,Nx),np.float32)

    u_part[-1,...] = u0
    ps = np.array(points).copy()
    
    for i in range(NZsplit):
        i1,i2 = i*Nz2, np.clip((i+1)*Nz2,0,Nz)
        # print u_part[-1,...]
        u_part, dn_part = bpm_3d_spheres((Nx,Ny,i2-i1+1),units = units,lam = lam,u0 = u_part[-1,...],
                                   points = ps,
                                   dn_inner = dn_inner, rad_inner = rad_inner,                    
                                   dn_outer = dn_outer, rad_outer = rad_outer)

        # shift points to correct position
        ps[:,-1] -= Nz2*units[-1]

        u[i1:i2,...] = u_part[1:,...]
        dn[i1:i2,...] = dn_part[1:,...]

    return u, dn



                                   
def test_3d_spheres():
    Nx, Nz = 256,512
    dx, dz = .1, 0.1

    lam = .5

    points =  [[dx*Nx/2.,dx*Nx/2.,5]]
    
    u, dn,r = bpm_3d_spheres((Nx,Nx,Nz),(dx,dx,dz),
                           points = points)


def test_split():
    Nx, Nz = 1024,1024
    Nx, Nz = 2048,1024

    #Nx, Nz = 512,512
    #Nx, Nz = 256,256
    Lx = 400
    Lz = 400

    dx, dz = .1, 0.2
  
    dx, dz = 1.*Lx/Nx, 1.*Lz/Nz

    lam = .5

    #points =  [[dx*Nx/2.,dx*Nx/2.,13.]]

    Np = 1000
    x = dx*np.random.uniform(.1*Nx,.9*Nx,Np)
    y = dx*np.random.uniform(.1*Nx,.9*Nx,Np)
    z = dz*np.random.uniform(0.3*Nz,.7*Nz,Np)
    points = np.array([x,y,z]).T

    #points =  [[dx*Nx/2.,dx*Nx/2.,dz*Nz/2.]]

    u,dn = bpm_3d_spheres_split((Nx,Nx,Nz),(dx,dx,dz),
                                   NZsplit = 16,
                           points = points)


if __name__ == '__main__':
    pass
    # from bpm.bpm_3d import bpm_3d_free
    
    # Nx, Nz = 1024,1024
    # Nx, Nz = 2048,1024

    # Nx, Nz = 512,512
    # # Nx, Nz = 256,256
    # Lx = 400
    # Lz = 400

    # dx, dz = 1.*Lx/Nx, 1.*Lz/Nz

    # x = np.linspace(-1,1,Nx)
    # Y,X = np.meshgrid(x,x,indexing="ij")
    # R = np.sqrt(X**2+Y**2)

    # u0_far = (R<.7)*np.cos(50*X)
    # u0_far = np.cos(2*np.pi*5*(R+.3*X**4))
    # u0_far = np.cos(2*np.pi*5*X)
    
    # u0 = bpm_3d_free((Nx,Nx),(dx,dx),Lz,u0 = u0_far).conjugate()
    

    # lam = .5

    # Np = 1000
    # x = dx*np.random.uniform(.1*Nx,.9*Nx,Np)
    # y = dx*np.random.uniform(.1*Nx,.9*Nx,Np)
    # z = dz*np.random.uniform(0.3*Nz,.5*Nz,Np)
    # points = np.array([x,y,z]).T

    # u,dn = bpm_3d_spheres_split((Nx,Nx,Nz),(dx,dx,dz),
    #                                NZsplit = 2,
    #                             u0 = u0,
    #                        points = points)
