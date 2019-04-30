#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:10:22 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import colors
from scattersim import Scatter

plt.rcParams['font.size']= 16
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['mathtext.fontset'] = 'cm'

rjtoau = 1/2150

a1, a2 = 1.0,1.1
e1, e2 = 0.0,0.8
r1, r2 = 1*rjtoau, 1*rjtoau

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
Rstar = 1/215

#We set up a set of q values which we will investigate

qvals = np.logspace(-4.1,4.1,100)

#We also define a set of bvals for which we will evaluate for each q

#We set up a range of b values
bmax = 0.01
bvals = np.linspace(-bmax,bmax,500)

bminarr11 = np.zeros(len(qvals))
bmaxarr11 = np.zeros(len(qvals))

bminarr12 = np.zeros(len(qvals))
bmaxarr12 = np.zeros(len(qvals))

bminarr21 = np.zeros(len(qvals))
bmaxarr21 = np.zeros(len(qvals))

bminarr22 = np.zeros(len(qvals))
bmaxarr22 = np.zeros(len(qvals))

qmask11   = np.full(len(qvals),False)
qmask12   = np.full(len(qvals),False)
qmask21   = np.full(len(qvals),False)
qmask22   = np.full(len(qvals),False)

#We then loop through all our q-values and save the points for which the upper
#and lower points in b for both of our orbital collision points.

scmask = np.zeros((len(qvals),len(bvals)),dtype=bool)
zz     = np.zeros((len(qvals),len(bvals)))

for i in range(len(qvals)):
    
    m1 = 1e-3
    m2 = qvals[i]*m1
    
    #We have to ensure that our mass ratio between the most massive planet and
    #the star is constant throughout the problem
    Mstar = np.max([m1,m2])*1e3
    
    p1data = np.array([a1,e1,m1,r1])
    p2data = np.array([a2,e2,m2,r2])    
    
    SC = Scatter(p1data,p2data,Mstar,Rstar,theta=theta)
    
    SC.scatter(b=bvals)
    
    #We should include the possibility of scattering the second planet into the 
    #star as well.
    
    mask11 = SC.scoll[:,0,0] & (SC.et1[:,0]<1)
    mask12 = SC.scoll[:,0,1] & (SC.et1[:,1]<1)
    
    mask21 = SC.scoll[:,1,0] & (SC.et2[:,0]<1)
    mask22 = SC.scoll[:,1,1] & (SC.et2[:,1]<1)
   
    scmask[i] = np.any([mask11,mask12,mask21,mask22],axis=0)
    
#    ax.fill(qp[mask11],bvals[mask11]/rjtoau,c='g',alpha=0.75)
#    ax.fill(qp[mask12],bvals[mask12]/rjtoau,c='g',alpha=0.75)
#    ax.fill(qp[mask21],bvals[mask21]/rjtoau,c='g',alpha=0.75)
#    ax.fill(qp[mask22],bvals[mask22]/rjtoau,c='g',alpha=0.75)

    if any(mask11):
        bminarr11[i] = bvals[mask11].min()
        bmaxarr11[i] = bvals[mask11].max()
        qmask11[i]   = True
    if any(mask12):
        bminarr12[i] = bvals[mask12].min()
        bmaxarr12[i] = bvals[mask12].max()
        qmask12[i]   = True               
    if any(mask21):
        bminarr21[i] = bvals[mask21].min()
        bmaxarr21[i] = bvals[mask21].max()
        qmask21[i]   = True
    if any(mask22):
        bminarr22[i] = bvals[mask22].min()
        bmaxarr22[i] = bvals[mask22].max()
        qmask22[i]   = True

plt.close('all')

fig, ax = plt.subplots(figsize=(10,6))    

xx,yy = np.meshgrid(qvals,bvals,indexing='ij')

zz[scmask] = 0

#We plot the region where Star-planet collision is possible
ax.fill_between(qvals,bminarr11/rjtoau,bmaxarr11/rjtoau,where=qmask11,color='g',alpha=0.75)
ax.fill_between(qvals,bminarr12/rjtoau,bmaxarr12/rjtoau,where=qmask12,color='g',alpha=0.75)
ax.fill_between(qvals,bminarr21/rjtoau,bmaxarr21/rjtoau,where=qmask21,color='g',alpha=0.75)
ax.fill_between(qvals,bminarr22/rjtoau,bmaxarr22/rjtoau,where=qmask22,color='g',alpha=0.75)

#We also make a legend handle
handle = plt.Rectangle((0, 0), 1, 1, fc="g",alpha=0.75,label=r'$e_{i,crit}<\tilde{e}_i<1$')

ax.set_xlabel('$q_p$')
ax.set_ylabel('$b\ [R_J]$')

ax.set_xscale('log')
ax.set_xlim(qvals.min(),qvals.max())
ax.set_ylim(-bmax/rjtoau,bmax/rjtoau)

#We add the current date

date = datetime.datetime.now()
        
datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
        
fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)

#We also add a table

celldata  = [[a1,e1,m1*1e3],[a2,e2,'$q m_1$']]
tabcol    = ['$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_J]$']
tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']

table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
          loc='top',cellLoc='center')

table.set_fontsize(10)

table.scale(1, 1.2)

yticks = ax.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax.legend(handles=[handle],prop={'size':13})
#plt.tight_layout()