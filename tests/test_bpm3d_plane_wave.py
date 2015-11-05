"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt


from bpm import bpm_3d

def test_plane(n_x_comp = 0, n0 = 1., n = None):
    """ propagates a plane wave freely
    n_x_comp is the tilt in x
    """
    Nx, Nz = 128,128
    dx, dz = .05, 0.05

    if n is None:
        n = n0
        
    lam = .5

    units = (dx,dx,dz)
    
    x = dx*np.arange(Nx)
    y = dx*np.arange(Nx)
    z = dz*np.arange(Nz)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")

    
    k_x = 1.*n_x_comp/(dx*(Nx-1.))

    
    k_z = np.sqrt(1.*n**2/lam**2-k_x**2)

    print np.sqrt(k_x**2+k_z**2)
    
    u_plane = np.exp(-2.j*np.pi*(k_z*Z+k_x*X))

    u = 0
    dn = (n-n0)*np.ones_like(Z)

    print n,n0, np.mean(dn)
    u, dn = bpm_3d((Nx,Nx,Nz),units= units, lam = lam,
                   n0 = n0,
                   dn = dn,
                   subsample = 2,
                   u0 = u_plane[0,...])

    # npt.assert_almost_equal(np.mean(np.abs(u_plane-u)**2),0,decimal = 2)
    return u, u_plane

if __name__ == '__main__':

    u1,u2 = test_plane(0,1.5)

    import pylab
    import seaborn
    pylab.figure(1)
    pylab.clf()
    pylab.plot(np.imag(u1)[:,64,64], label="bpm")
    pylab.draw()
    pylab.plot(np.imag(u2)[:,64,64], label="analy")
    pylab.draw()
    pylab.legend()
    pylab.show()
    # test_plane(1)
    # test_plane(2)
    
