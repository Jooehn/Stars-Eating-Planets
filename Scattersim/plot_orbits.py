#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:00:06 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from labellines import labelLines

def get_orbit_r(ang):
    
    rvals = []
    for i in range(len(pdata)):
        a, e = pdata[i]
    
        #We work out the semi-latus rectum
        if e<1:
            p = a*(1-e**2)       
        elif e==1:
            p = 2*a
        #This yields the following r for our orbit
        r = p/(1+e*np.cos(ang))
        rvals.append(r)

    return rvals

def plot_orbit(pdata):
    """Plots the circular and eccentric orbits and marks the point of crossing."""
    
    a1, e1 = pdata[0]
    a2, e2 = pdata[1]
    a3, e3 = pdata[2]
    
    ang = np.linspace(0,2*np.pi,1000)
    
    r1, r2, r3 = get_orbit_r(ang)
    
    x1 = r1*np.cos(ang)
    y1 = r1*np.sin(ang)
    x2 = r2*np.cos(ang)
    y2 = r2*np.sin(ang)
    x3 = r3*np.cos(ang)
    y3 = r3*np.sin(ang)
    
    fig, ax = plt.subplots(figsize=(8,8))
    
#    ax.plot(x1,y1,'b-',label='$\mathrm{Orbit\ 1}$')
#    ax.plot(x2,y2,'r-',label='$\mathrm{Orbit\ 2}$')
#    ax.plot(x3,y3,'g-',label='$\mathrm{Orbit\ 3}$')
    line1, = ax.plot(x1,y1,'b-',label='$e=0$')
    line2, = ax.plot(x2,y2,'r-',label='$e=0.8$')
    line3, = ax.plot(x3,y3,'g-',label='$e=1$')
    ax.plot(0,0,marker='+',color='tab:gray',ms=10)
    ax.set_aspect('equal')
    
    xmax = int(np.ceil(np.amax(np.absolute([x1,x2,x3]))))
    ymax = int(np.ceil(np.amax(np.absolute([x1,x2,x3]))))
    
    xymax = 4
    
    ax.set_xlim(-xymax,xymax)
    ax.set_ylim(-xymax,xymax)
    ax.set_yticks(np.arange(-xymax,xymax+1,1,dtype=int))
    ax.set_xlabel('$x\ \mathrm{[AU]}$')
    ax.set_ylabel('$y\ \mathrm{[AU]}$')
    
    labelLines([line1,line2,line3],fontsize=14,xvals=[0,-1.6,1])
#    ax.legend(prop={'size':13})
    
rjtoau = 1/2150
metoms = 1/332946
a1, a2, a3 = 2.0,2.0,2.0
e1, e2, e3 = 0.0,0.8,1.0
p1data = np.array([a1,e1])
p2data = np.array([a2,e2])
p3data = np.array([a3,e3])

pdata = np.vstack([p1data,p2data,p3data])

#We save the plot
date = datetime.datetime.now()

plot_orbit(pdata)
os.chdir(os.getcwd()+'/../../Report/Figures')
plt.savefig('orbit_ecc_{0}_{1}_{2}.pdf'.format(date.day,date.month,date.year))
plt.close('all')