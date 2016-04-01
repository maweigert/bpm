"""


mweigert@mpi-cbg.de

"""
from numpy import *



lam = .5
k0 = 2*pi/lam


N = 256
dx = .1
x = dx*(arange(-N/2,N/2)+.5)
kx = 2.*pi*fft.fftfreq(N,dx)
kx_pad = 2.*pi*fft.fftfreq(2*N,dx)

z = dx*arange(N)
Z,X = meshgrid(z,x, indexing = "ij")
R = sqrt((Z-4)**2+X**2)
dn = .1*(R<2.)

H0 = sqrt(0.j+k0**2-kx**2).conjugate()
H0_pad = sqrt(0.j+k0**2-kx_pad**2).conjugate()

outsideInds = isnan(H0)

def _pad(u, val = 0., tapwidth = 0):
    if tapwidth ==0:
        return pad(u,(N/2,N/2),mode = "constant",
               constant_values = (val,val))



def _ipad(u):
    return u[N/2:-N/2]

def prop(u,z, padme = True, padval = 0.):
    _H = exp(-1.j*z*H0)
    _H_pad = exp(-1.j*z*H0_pad)
    _H[outsideInds] = 0.

    if padme:
        f = fft.fftn(_pad(u,val =padval))
        f *= _H_pad
        return _ipad(fft.ifftn(f))
    else:
        f = fft.fftn(u)
        f*= _H
        return fft.ifftn(f)

def propa(u0,dz,Nz,dn, padme = True):
    u = zeros((Nz,len(u0)), complex64)
    u[0] = u0
    for i in xrange(Nz-1):
        u[i+1] = prop(u[i],dz, padme = padme,padval = 0*exp(-1.j*i*dz*k0))
        u[i+1] *= exp(-1.j*dn[i+1])
    return u

def gaussian(r,z,lam=.5,w0=.6):
    k  =2.*pi/lam
    zr = pi*w0**2/lam
    w  = w0*sqrt(1+(z/zr)**2)
    R = z*(1+(zr/z)**2)
    gouy = arctan(z/zr)
    return w0/w*exp(-r**2/w**2)*exp(-1.j*(k*z+k*r**2/2./R-gouy))



u0 = ones(N,complex64)
u0 = gaussian(abs(x),-dx*N/2.)

u1 = propa(u0,dx,N,1.*dn, padme = False)
u2 = propa(u0,dx,N,1*dn, padme = True)