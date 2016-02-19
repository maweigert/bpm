"""
some tests to see if we can improve the classical bpm via svd decomposition
(like WPM)
"""
import numpy as np
import fbpca


def first_svd_comp(h, n_comps=1):
    u,s,v = np.linalg.svd(h)

    comps = [(np.sqrt(s[i])*u[:,i],np.sqrt(s[i])*v[i,:]) for i in range(n_comps)]
    return comps

def first_svd_comp_fb(h, n_comps=1):
    u,s,v = fbpca.pca(h,k = n_comps, raw = True)

    comps = [(np.sqrt(s[i])*u[:,i],np.sqrt(s[i])*v[i,:]) for i in range(n_comps)]
    return comps


def prop_standard(size,units,n):
    lam = .5

    k0 = 2.*np.pi/lam

    Nx, Nz = size
    dx, dz = units
    k = 2.*np.pi*np.fft.fftfreq(Nx,dx)
    n0 = 1.

    u = np.ones((Nz,Nx),np.complex64)

    for i in range(Nz-1):
        f = np.exp(dz*1.j*k0*(N[i+1]-n0))
        g = np.exp(dz*1.j*np.sqrt(0.j+n0**2*k0**2-k**2))

        u[i+1] = f*np.fft.ifftn(g*np.fft.fftn(u[i]))

    return u



def prop_mean(size,units,n):
    lam = .5

    k0 = 2.*np.pi/lam

    Nx, Nz = size
    dx, dz = units
    k = 2.*np.pi*np.fft.fftfreq(Nx,dx)

    u = np.ones((Nz,Nx),np.complex64)

    for i in range(Nz-1):
        n0 = np.mean(n[i+1])
        f = np.exp(dz*1.j*k0*(n[i+1]-n0))
        g = np.exp(dz*1.j*np.sqrt(0.j+n0**2*k0**2-k**2))


        u[i+1] = f*np.fft.ifftn(g*np.fft.fftn(u[i]))

    return u



def prop_svd(size,units,n, n_comps = 10):
    lam = .5

    k0 = 2.*np.pi/lam

    Nx, Nz = size
    dx, dz = units
    k = 2.*np.pi*np.fft.fftfreq(Nx,dx)

    u = np.ones((Nz,Nx),np.complex64)

    for i in range(Nz-1):
        print i


        K, N = np.meshgrid(k,n[i+1])
        H = np.exp(dz*1.j*np.sqrt(0.j+N**2*k0**2-K**2))

        res = np.zeros(Nx,np.complex64)

        comps = first_svd_comp_fb(H,n_comps = n_comps)

        for f,g in comps:
            res += f*np.fft.ifftn(g*np.fft.fftn(u[i]))

        u[i+1] = res

    return u



if __name__ == '__main__':
    #some refractive index dist

    Nx, Nz = 256, 400
    dx = .1

    x = dx*(np.arange(0,Nx)-Nx/2.)
    z = dx*(np.arange(0,Nz)-Nz/3.)
    Z, X = np.meshgrid(z,x,indexing="ij")

    R = np.sqrt(X**2+Z**2)
    N = 1.+.4*np.exp(-.1*R**2)
    N = 1.+.05*(R<4.)



    u1 = prop_standard((Nx,Nz),(dx,)*2,N)
    u2 = prop_mean((Nx,Nz),(dx,)*2,N)
    u3 = prop_svd((Nx,Nz),(dx,)*2,N)