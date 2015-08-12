import numpy as np

    
from bpm import bpm_3d


def test_simple_3d():                        

    Nx, Nz = 128,256
    dx, dz = .1, 0.1

    lam = .5

    units = (dx,dx,dz)
    rad = 2.

    x = dx*np.arange(-Nx/2,Nx/2)
    z = dz*np.arange(-Nz/4,3*Nz/4)
    Z,Y,X = np.meshgrid(z,x,x,indexing="ij")
    R = np.sqrt(X**2+Y**2+Z**2)
    dn = .1*(R<2.)
    
    u1, dn1 = bpm_3d((Nx,Nx,Nz),
                     units= units,
                     lam = lam,
                     dn = dn,
                     return_scattering = False )
 
    u2, dn2 = bpm_3d((Nx,Nx,Nz),
                     units= units,
                     lam = lam,
                     dn = dn,
                     n_volumes = 2,
                     return_scattering = False )
    
    return u1,u2

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

def test_plane(tilt = 0, n0 = 1.):
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

    
    k_x = 1.*tilt/(dx*(Nx-1.))

    
    k_z = np.sqrt(1.*n0**2/lam**2-k_x**2)
    u_plane = np.exp(2.j*np.pi*(k_z*Z+k_x*X))

    u = 0

    u, dn = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
                   n0 = n0,
                   u0 = u_plane[0,...])
    
    print np.mean(np.abs(u_plane-u)**2)
    return u, u_plane

# def test_slit():
#     Nx, Nz = 128,128
#     dx, dz = .05, 0.05

#     lam = 0.5

#     units = (dx,dx,dz)

    
    
#     x = np.linspace(-1,1,Nx)
#     y = np.linspace(-1,1,Nx)
#     Y,X = np.meshgrid(y,x,indexing="ij")

#     R = np.hypot(Y,X)

#     u0 = 1.*(R<.5) 

    
#     u, dn, p = bpm_3d((Nx,Nx,Nz),units= units,
#                       lam = lam,
#                       u0 = u0,
#                       dn = np.zeros((Nz,Nx,Nx)),
#                       subsample = 1,
#                       return_scattering = True )

#     return u, dn, p


# def test_sphere():
#     Nx, Nz = 128,128
#     dx, dz = .05, 0.05

#     lam = .5

#     units = (dx,dx,dz)
    
#     x = Nx/2*dx*np.linspace(-1,1,Nx)
#     y = Nx/2*dx*np.linspace(-1,1,Nx)
    
#     x = dx*np.arange(-Nx/2,Nx/2)
#     y = dx*np.arange(-Nx/2,Nx/2)
#     z = dz*np.arange(0,Nz)
#     Z,Y,X = np.meshgrid(z,y,x,indexing="ij")
#     R = np.sqrt(X**2+Y**2+(Z-3.)**2)
#     dn = .05*(R<1.)
    
#     u, dn, p = bpm_3d((Nx,Nx,Nz),units= units,
#                       lam = lam,
#                       dn = dn,
#                       subsample = 1,
#                       n_volumes = 1,
#                       return_scattering = True )

    
#     print np.sum(np.abs(u[1:,...]))
#     return u, dn,p

# def test_compare():
#     Nx, Nz = 128,256
#     dx, dz = .05, 0.05

#     lam = .5

#     units = (dx,dx,dz)

    
    
#     x = Nx/2*dx*np.linspace(-1,1,Nx)
#     y = Nx/2*dx*np.linspace(-1,1,Nx)
    
#     x = dx*np.arange(-Nx/2,Nx/2)
#     y = dx*np.arange(-Nx/2,Nx/2)
#     z = dz*np.arange(0,Nz)
#     Z,Y,X = np.meshgrid(z,y,x,indexing="ij")
#     R = np.sqrt(X**2+Y**2+(Z-3.)**2)
#     dn = .05*(R<1.)
    
#     u1, dn1, p1 = bpm_3d((Nx,Nx,Nz),units= units,
#                       lam = lam,
#                       dn = dn,
#                       subsample = 1,
#                       n_volumes = 1,
#                       return_scattering = True )

#     u2, dn2, p2 = bpm_3d_old((Nx,Nx,Nz),units= units,
#                       lam = lam,
#                       dn = dn,
#                       return_scattering = True )
    
#     return u1, u2



if __name__ == '__main__':
    # u1, u2 = test_simple_3d()

    # test_speed()

    u1, u2 = test_plane(tilt= 1, n0 = 1.2)
