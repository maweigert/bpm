

from bpm.utils import StopWatch, absPath
from bpm.bpm_3d_spheres import bpm_3d_spheres


import sys
sys.path.append("/Users/mweigert/python/mie_scatt")
from bhmie_code import bhmie


from numpy import *

def mie_eff(r, lam = .5,dn= .1):
    S1,S2,Qext, Qsca,Qback,gsca = bhmie(2*pi*r/lam,1.+dn,128)
    return Qsca


def bpm_eff(r,lam= .5, dn= .05):
    u, dn, p = bpm_3d_spheres((256,)*3,units= (.1,)*3,
                              points = [[12.4,12.4,12.4]],
                              lam = lam,
                              dn_inner = 0,rad_inner=0,
                              dn_outer=dn, rad_outer=r,
                              return_scattering = True )
    return p[-1]/r**2/pi


if __name__ == '__main__':

    lam = .5
    
    dns = linspace(.01,.2,20)

    rs = linspace(1.,10,50)
    
    p_mie = array([[mie_eff(r,lam,dn) for r in rs] for dn in dns])

    p_bpm = array([[bpm_eff(r,lam,dn) for r in rs] for dn in dns])
