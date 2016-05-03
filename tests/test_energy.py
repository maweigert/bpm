"""
check whether we conserve energy along z
"""

import bpm
import numpy as np
import pylab

if __name__ == '__main__':

    N = 256
    dn = .1*np.random.uniform(-1,1,(N,)*3)

    #plane wave through empty space  and through dn 
    u0 = bpm.bpm_3d((N,)*3,(.1,)*3)
    u = bpm.bpm_3d((N,)*3,(.1,)*3,dn = dn)


    pylab.ioff()
    pylab.plot(np.mean(np.abs(u0),(1,2)))
    pylab.plot(np.mean(np.abs(u),(1,2)))

    pylab.show()
                   

    
    

    
