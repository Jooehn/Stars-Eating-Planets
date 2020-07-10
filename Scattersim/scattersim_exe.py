#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:39:17 2019

@author: John Wimarsson

Executable file for the scattersim class used in the Master thesis
'Stars Eating Planets'
"""

from scattersim import *

#We set up the data for the run

#The data for the two planets

plt.close('all')

rjtoau = 1/2150
metoms = 1/332946
a1, a2 = 1.0,0.9
e1, e2 = 0.0,0.85
m1, m2 = 300*metoms, metoms
#m1, m2 = 0.0004218897917605734, 0.00048216071492819887
p1data = np.array([a1,e1,m1])
p2data = np.array([a2,e2,m2])

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
Mstar = 1

SC = Scatter(p1data,p2data,Mstar,theta=theta)

#Now, we can call the functions

#The orbits can be plotted using the following function 

SC.plot_orbit()

#We then perform a single scattering with an impact parameter b

#b = 0.1
#SC.find_bmax()
#b = 0.25*SC.bmax
#SC.scatter(b = np.array([b]))
#The corresponding vector triangle is given by

#SC.plot_vectri(2,0)
#print(np.rad2deg(SC.defang))

#We can also plot the resulting orbital elements after a scatterings with a set
#of bvals given in an interval
bmax = SC.find_bmax()
bvals = np.linspace(-bmax,bmax,1000)
SC.plot_new_orb(bvals=bvals,idx=0)