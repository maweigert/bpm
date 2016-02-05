"""
some tests to see if we can improve the classical bpm via svd decomposition
(like WPM)
"""
import numpy as np
# from sklearn.decomposition import TruncatedSVD
# trunk = TruncatedSVD(n_components=1)
# def first_svd_comp(x):
#     trunk.fit(x)


def first_svd_comp(h):
    u,s,v = np.linalg.svd(h)
    u0 = np.sqrt(s[0])*u[:,0]
    v0 = np.sqrt(s[0])*v[0,:]
    return u0,v0, np.outer(u0,v0)

def phase(x):
    return np.imag(np.log(x))

if __name__ == '__main__':
    Nx = 256


    x = np.linspace(-1,1,Nx)
    k = np.linspace(-1,1,Nx+1)[:-1]


    #some refractive index dist
    n = 1.+.1*np.exp(-10*(x-.2)**2)

    K,N = np.meshgrid(k,n,indexing="ij")

    # the spectral weight over which to minimize
    u0 = 1.+0*k
    u0 = np.exp(-10*k**2)
    # u0 *= 0
    # u0[Nx/2] = 1.
    #
    U0 = np.outer(u0,np.ones(Nx))

    # the actual propagator as in WPM
    P = U0*np.exp(1.j*np.sqrt(N**2-K**2))

    # the bpm approx
    n0 = 1.
    sn0 =  np.exp(1.j*(n-n0))
    sk0 = u0*np.exp(1.j*np.sqrt(n0**2-k**2))
    P0 = U0*np.exp(1.j*(np.sqrt(n0**2-K**2)+(N-n0)))

    assert np.allclose(P0,np.outer(sk0,sn0))

    # the mean bpm approx
    nm = np.mean(n)
    P1 = U0*np.exp(1.j*(np.sqrt(nm**2-K**2)+(N-nm)))
    sn1 =  np.exp(1.j*(n-nm))
    sk1 = u0*np.exp(1.j*np.sqrt(nm**2-k**2))

    assert np.allclose(P1,np.outer(sk1,sn1))

    # the svd approx
    sk2,sn2,P2 = first_svd_comp(P)

