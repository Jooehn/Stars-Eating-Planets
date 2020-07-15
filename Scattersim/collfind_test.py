#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:07:51 2019

@author: jooehn
"""

from scattersim import *

#We set up the data for the run

#The data for the two planets

plt.close('all')

rjtoau = 1/2150
metoms = 1/332946
a1, a2 = 1,1.1
e1, e2 = 0.0,0.8
m1, m2 = 300*metoms, metoms
p1data = np.array([a1,e1,m1])
p2data = np.array([a2,e2,m2])

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = np.random.uniform(0,360)
Mstar = 1
Rstar = 1/215

SC = Scatter(p1data,p2data,Mstar,theta=theta)

SC.collfinder(1000)