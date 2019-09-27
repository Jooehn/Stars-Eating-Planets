#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:36:41 2019

@author: John Wimarsson

A script that investigates for which combinations of semi-major axis and 
eccentricity that can lead to at least one star-planet collision and how
the fraction of combinations in our parameter space that lead to star-planet
collision varies with mass of the secondary component.
"""

from scattersim import *

plt.close('all')

#We first set up our grid of a and e values

avals = np.arange(1,50+1,1)
evals = np.arange(0,1,0.1)

#aa, ee = np.meshgrid(avals,evals)

rjtoau = 1/2150
metoms = 1/332946
a1 = 1.0
e1 = 0.0
m1 = 300*metoms

#We loop through our parameter space and perform a scattering for each system
#where a1, e1 are kept static and a2, e2 vary

for i in len(avals):
    
    a2 = avals[i]
    
    for j in len(evals):
        
        e2 = evals[j]
        
        p1data = np.array([a1,e1,m1])
        p2data = np.array([a2,e2,m2])
        
        SC = Scatter(p1data,p2data,M_s,R_s)
        
        bmax = SC.find_bmax()
        
        bvals = np.linspace(-bmax,bmax,100)
        
        SC.scatter(bvals)
        
        f_eje = SC.eject.sum()
        f_scp = SC.scoll.sum()
        f_mer = SC.merger.sum()
        
        