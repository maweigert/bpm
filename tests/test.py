"""the main method for beam propagation in media with coated spheres"""

import numpy as np

from bpm import bpm_3d


if __name__ == '__main__':
    
    N = 128
    dx = .1

    u,_ = bpm_3d((N,)*3,(dx,)*3,lam = .5)

    
