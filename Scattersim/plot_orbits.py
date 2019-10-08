#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:00:06 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt

def get_orbit_r(ang):
    
    a1, e1 = pdata[0]
    a2, e2 = pdata[1]
    a3, e3 = pdata[2]
    
    #We work out the semi-latus rectum
    p1 = a1*(1-e1**2)
    p2 = a2*(1-e2**2)
    p3 = a3*(1-e3**2)
    
    #This yields the following r for our orbit
    r1 = p1/(1+e1*np.cos(ang))
    r2 = p2/(1+e2*np.cos(ang))
    r3 = p3/(1+e3*np.cos(ang))
    
    return r1,r2,r3

def plot_orbit(pdata):
    """Plots the circular and eccentric orbits and marks the point of crossing."""
    
    a1, e1 = pdata[0]
    a2, e2 = pdata[1]
    a3, e3 = pdata[2]
    
    ang = np.linspace(0,2*np.pi,1000)
    
    r1, r2, r3 = get_orbit_r(ang)
    
    x1 = r1*np.cos(ang+np.pi)
    y1 = r1*np.sin(ang+np.pi)
    x2 = r2*np.cos(ang)
    y2 = r2*np.sin(ang)
    x3 = r3*np.cos(ang)
    y3 = r3*np.sin(ang)
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.plot(x1,y1,'b-',label='$\mathrm{Orbit\ 1}$')
    ax.plot(x2,y2,'r-',label='$\mathrm{Orbit\ 2}$')
    ax.plot(x3,y3,'g-',label='$\mathrm{Orbit\ 3}$')
    ax.plot(0,0,marker='+',color='tab:gray',ms=10)
    ax.set_aspect('equal')
    
    xmax = int(np.ceil(np.amax(np.absolute([x1,x2,x3]))))
    ymax = int(np.ceil(np.amax(np.absolute([x1,x2,x3]))))
    
    ax.set_xlim(-xmax-1,xmax+1)
    ax.set_ylim(-ymax-1,ymax+1)
    ax.set_yticks(np.arange(-ymax,ymax+1,1,dtype=int))
    ax.set_xlabel('$x\ \mathrm{[AU]}$')
    ax.set_ylabel('$y\ \mathrm{[AU]}$')
    
    ax.legend(prop={'size':13})
    
rjtoau = 1/2150
metoms = 1/332946
a1, a2, a3 = 1.0,0.3,2.0
e1, e2, e3 = 0.0,0.9,0.0
p1data = np.array([a1,e1])
p2data = np.array([a2,e2])
p3data = np.array([a3,e3])

pdata = np.vstack([p1data,p2data,p3data])

plot_orbit(pdata)