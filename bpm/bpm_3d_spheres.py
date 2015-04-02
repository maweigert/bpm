"""the main method for beam propagation in media with coated spheres"""

import numpy as np

from volust.volgpu import OCLArray, OCLProgram

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
                   dn_inner = 0.04, rad_inner = 3.,                   
                   dn_outer = 0.02, rad_outer = 4.,
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

    u_part[-1,...] = u0
    
    for i in range(NZsplit):
        i1,i2 = i*Nz2, np.clip((i+1)*Nz2,0,Nz)
        # print u_part[-1,...]
        u_part, _ = bpm_3d_spheres((Nx,Ny,i2-i1+1),units = units,lam = lam,u0 = u_part[-1,...],
                                   points = points,
                                   dn_inner = dn_inner, rad_inner = rad_inner,                    
                                   dn_outer = dn_outer, rad_outer = rad_outer)

        u[i1:i2,...] = u_part[1:,...]

    return u



                                   
def test_3d_spheres():
    Nx, Nz = 256,512
    dx, dz = .1, 0.1

    lam = .5

    points =  [[dx*Nx/2.,dx*Nx/2.,5]]
    
    u, dn,r = bpm_3d_spheres((Nx,Nx,Nz),(dx,dx,dz),
                           points = points)

if __name__ == '__main__':
    Nx, Nz = 512,512
    dx, dz = .05, 0.05

    lam = .5

    points =  [[dx*Nx/2.,dx*Nx/2.,13.]]
    
    u = bpm_3d_spheres_split((Nx,Nx,Nz),(dx,dx,dz),
                                   NZsplit = 4,
                           points = points)
