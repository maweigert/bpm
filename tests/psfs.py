"""the main method for beam propagation in media with coated spheres"""

import numpy as np

from bpm.psf_integrals import psf_debye, psf_debye_gauss


if __name__ == '__main__':
    
    N = 256
    dx = .1
    NA = .7

    u,ex,ey,ez = psf_debye((N,)*3,(dx,)*3,lam = .5,NAs=[0.,NA])
    u2,ex2,ey2,ez2 = psf_debye_gauss((N,)*3,(dx,)*3,lam = .5,NAs=[0.,NA], sig = 1./np.sqrt(2))
    
