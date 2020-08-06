#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:29:22 2020

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from scattersim import Scatter
from plotfuncs import *

def cross_check():
    """Checks if the two specified orbits will ever cross"""
    
    if a1>a2:
        if (a1*(1-e1)) > (a2*(1+e2)):
            return False
        elif np.allclose((a1*(1-e1)),(a2*(1+e2))):
            return False
    elif a1<a2:
        if (a1*(1+e1)) < (a2*(1-e2)):
            return False
        elif np.allclose((a1*(1+e1)),(a2*(1-e2))):
            return False
        
    return True

def get_isec():
    """Obtains the angle of intersection of the orbits given the planet data"""
    
    #We go over all angles in an interval to check the distances to the 
    #current position of the planets. If they are equal we have a collision
    #We use steps of 0.1 degrees
    
    ang = np.deg2rad(np.arange(.1,360+.1,.1))
    
    phi1 = ang
    phi2 = ang-theta
    
    r1 = a1*(1-e1**2)/(1+e1*np.cos(phi1))
    r2 = a2*(1-e2**2)/(1+e2*np.cos(phi2))
    
    #We then calculate the difference between these values
    rdiff = r1-r2
    
    #If these are equal to zero at any point, the corresponding angle
    #corresponds to a crossing. Otherwise, we look for the point where
    #the differences changes sign. The true angle value will then be
    #in between the values for which the sign change occurs.
    
    phic = np.zeros(2)
    
    done = False
    crossing = False
    if any(np.isclose(rdiff,0)):
        cidx = np.where(np.isclose(rdiff,0))[0]
        if np.size(cidx)==2:
            phic[0] += ang[cidx[0]]#,ang[cidx[0]]
            phic[1] += ang[cidx[1]]#,ang[cidx[1]]
            done = True
            crossing = True
    elif not done:
        #We find the sign for each element and use np.roll to find the two points
        #where it changes
        rdsign = np.sign(rdiff)
        sz = rdsign == 0
        while sz.any():
            rdsign[sz] = np.roll(rdsign, 1)[sz]
            sz = rdsign == 0
        schange = ((np.roll(rdsign, 1) - rdsign) != 0).astype(int)
        #We set the first element to zero due to the circular shift of
        #numpy.roll
        schange[0] = 0
        #Finally we check where the array is equal to one and extract the
        #corresponding indices
        scidx = np.where(schange)[0]
        
        if np.size(scidx)==2:
            cidx = scidx
            crossing = True
        elif np.size(scidx)==0:
            crossing = False
        else:
            cidx = np.append(cidx,scidx)
            crossing = True
            
        #We now peform a linear extrapolation 
        if crossing:
            k1 = (r1[cidx]-r1[cidx-1])/(ang[cidx]-ang[cidx-1])
            m1 = r1[cidx-1]-k1*ang[cidx-1]
            k2 = (r2[cidx]-r2[cidx-1])/(ang[cidx]-ang[cidx-1])
            m2 = r2[cidx-1]-k2*ang[cidx-1]
            
            #We then have everything we need to find our crossing angles
            phic = (m2-m1)/(k1-k2)
    
    #We also compute and save the distance between the orbits crossings and
    #the host star
    # rc = (a1*(1-e1**2))/(1+e1*np.cos(phic))
    return crossing

def calc_delta(a1,a2):
    
    Rhillm = ((m1+m2)/(3*Mstar))**(1/3)*0.5*(a1+a2)
    
    delta = (a2-a1)/Rhillm
    
    return delta

plt.close('all')

#Constants
N_sys = int(1e2)
rjtoau = 1/2150
metoms = 1/332946

#We want to make N_sys crossing orbits and set up containers for the params
#1. semi-major axis
#2. eccentricity
#3. the angle between the eccentricity vectors of the two orbits

crossarr = np.zeros((N_sys,5))

#System props
m1, m2 = 300*metoms, 300*metoms
Mstar = 1

N = 0
s = 0
while N<N_sys:
    
    e1 = np.random.uniform(0,1)
    e2 = np.random.uniform(0,1)

    ai = np.random.uniform(0.1,20)
    aj = np.random.uniform(0.1,20)

    a1 = min(ai,aj)
    a2 = max(ai,aj)

    theta = np.random.uniform(0,2*np.pi)
    
    s += 1
    
    #First we check if two orbits with theta = 0 can cross
    if not cross_check:
        continue
    
    #Next we involve the theta angle
    if not get_isec():
        continue
    
    #If we get this far, we have a crossing orbit and can save the parameters
    crossarr[N,0] = a1
    crossarr[N,1] = e1
    crossarr[N,2] = a2
    crossarr[N,3] = e2
    crossarr[N,4] = theta
    
    N+=1
    
fig, ax = plt.subplots()

de = abs(crossarr[:,3] - crossarr[:,1])
da = (crossarr[:,2]-crossarr[:,0])/crossarr[:,2]
delta = calc_delta(crossarr[:,0],crossarr[:,2])

ax.scatter(delta,de,s=5,c=crossarr[:,4],cmap='viridis')
ax.set_xlim(0,25)
# ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.set_xlabel(r'$\Delta$')
# ax.set_xlabel('$(a_2 - a_1)/a_2$')
ax.set_ylabel('$e_2-e_1$')