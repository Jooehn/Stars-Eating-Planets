#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:34:35 2019

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
import os

plt.close('all')

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
metoms = 1/332946
mjtoms = 300*metoms
rjtoau = 1/2150

a1, a2 = 1.0,0.9
e1, e2 = 0.0,0.9
m1, m2 = 300*metoms, metoms

p1data = np.array([a1,e1,m1])
p2data = np.array([a2,e2,m2])    

#We set up a set of qstar values which we will investigate by boosting the mass
#of the star for each iteration.

Msvals = np.logspace(-1,1,300)

#Set up a few empty containers

bmvals = np.zeros(len(Msvals))

#We then loop through all our q-values and save the points for which the upper
#and lower points in b for both of our orbital collision points.

#fig, ax = plt.subplots(figsize=(10,6))
fig, ax = plt.subplots()

for i in range(len(Msvals)):
    
    #For each mass combination we perform a scattering and plot the result
    Mstar = Msvals[i]
    
    try:
        SC = Scatter(p1data,p2data,Mstar,theta=theta)
    except Exception:
        continue
    
    #We define a set of bvals for which we will evaluate for each qstar
    bmax  = SC.find_bmax(fac=0.01,step=0.001)
#    bmvals[i] = bmax/SC.Rhill_mut
    bvals = np.linspace(-bmax,bmax,int(1e4)) 
    
    #We perform the scatterings
    SC.scatter(b=bvals)
    
    #Next we find which planets have critically interacted with the host star
    #as well as which ones have collided
    
    mask11 = SC.scoll[:,0,0]
    mask12 = SC.scoll[:,0,1]
    
    mask21 = SC.scoll[:,1,0]
    mask22 = SC.scoll[:,1,1]
    
    #Further, we check the outcome of each scattering
    
    ap = np.asarray([Msvals[i]]*len(bvals))

    ds = 5
    
    scim = ax.scatter(ap[mask11],bvals[mask11]/bmax,c=SC.et1[:,0][mask11],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(ap[mask12],bvals[mask12]/bmax,c=SC.et1[:,1][mask12],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(ap[mask21],bvals[mask21]/bmax,c=SC.et2[:,0][mask21],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(ap[mask22],bvals[mask22]/bmax,c=SC.et2[:,1][mask22],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    
    ax.scatter(ap[mask12],bvals[mask12]/bmax,c=SC.et1[:,0][mask12],s=10,alpha=0,hatch='/')
    ax.scatter(ap[mask22],bvals[mask22]/bmax,s=10,alpha=0,hatch='/')

#We also make a legend handle
ghand = plt.Rectangle((0, 0), 1, 1, fc="darkgreen",label=r'$e_{i,crit}<\tilde{e}_i<1$')
hhand1 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',label='$\mathrm{Orbit\ crossing\ A}$')
hhand2 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',hatch='/',label='$\mathrm{Orbit\ crossing\ B}$')

ax.set_xlabel(r'$M_\star\ \mathrm{[M}_\odot]$')
ax.set_ylabel(r'$b\ [b_\mathrm{max}]$')

ax.set_xlim(Msvals.min(),Msvals.max())
ax.set_ylim(-1,1)
ax.set_xscale('log')

cbar = fig.colorbar(scim,ax=ax,ticks=np.arange(0.8,1+0.05,0.05))
cbar.ax.set_ylabel(r'$\tilde{e}$')

cbox = cbar.ax.get_position() # get the original position 
cpos = [cbox.x0 + 0.06, cbox.y0,  cbox.width, cbox.height] 
cbar.ax.set_position(cpos) # set a new position

box = ax.get_position() # get the original position 
pos = [box.x0 + 0.06, box.y0,  box.width, box.height] 
ax.set_position(pos) # set a new position

#We add the current date

date = datetime.datetime.now()
        
datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
        
#fig.text(0.88,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)

#We also add a table

celldata  = [[a1,e1,int(m1/metoms)],[a2,e2,int(m2/metoms)]]
tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']

table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
          loc='top',cellLoc='center')

table.set_fontsize(11.8) 
table.scale(1, 1.44)

yticks = ax.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax.legend(handles=[ghand,hhand1,hhand2],prop={'size':13})

os.chdir(os.getcwd()+'/../../Results/Zucc plots')
plt.savefig('zucc_qstar_plot.png',dpi=300)
plt.close('all')