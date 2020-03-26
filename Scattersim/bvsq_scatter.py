"""
Created on Tue Apr 23 16:10:22 2019

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

rjtoau = 1/2150

a1, a2 = 1.0,0.9
e1, e2 = 0.0,0.9

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
Rstar = 1/215
Mstar = 1

#We set up a set of q values which we will investigate by setting up the values
#of our planets for each iteration.

earthtosunm = 1/332946

N_mvals = 300
qvals = np.logspace(np.log10(1/300),np.log10(300),N_mvals)

#The total mass of the planets is mtot
mtot = 301
#We solve mtot = m1+m2 = m1+m1*qp for m1 and calculate our masses
m1vals = mtot/(1+qvals)
m2vals = qvals*m1vals

m1vals *= earthtosunm
m2vals *= earthtosunm

#Set up a few empty containers
bmvals    = np.zeros(len(qvals))

#We then loop through all our q-values and save the points for which the upper
#and lower points in b for both of our orbital collision points.

fig, ax = plt.subplots(figsize=(8,6))

for i in range(len(qvals)):
    
    #For each mass combination we perform a scattering and plot the result
    
    m1 = m1vals[i]
    m2 = m2vals[i]
    
    p1data = np.array([a1,e1,m1])
    p2data = np.array([a2,e2,m2])    
    
    SC = Scatter(p1data,p2data,Mstar,theta=theta)
    
    #We define a set of bvals for which we will evaluate for each qstar
    bmax  = SC.find_bmax(fac=0.01,step=0.001)
    bmvals[i] = bmax/SC.Rhill_mut
    bvals = np.linspace(-bmax,bmax,int(1e4)) 
    
    #We perform the scatterings
    try:
        SC.scatter(b=bvals)
    except ValueError:
        pass
    
    #Next we find which planets have critically interacted with the host star
    #as well as which ones have collided
    
    mask11 = SC.scoll[:,0,0]
    mask12 = SC.scoll[:,0,1]
    
    mask21 = SC.scoll[:,1,0]
    mask22 = SC.scoll[:,1,1]
    
    qp = np.asarray([qvals[i]]*len(bvals))

    ds = 5
    scim = ax.scatter(qp[mask11],bvals[mask11]/bmax,c=SC.et1[:,0][mask11],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(qp[mask12],bvals[mask12]/bmax,c=SC.et1[:,1][mask12],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(qp[mask21],bvals[mask21]/bmax,c=SC.et2[:,0][mask21],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(qp[mask22],bvals[mask22]/bmax,c=SC.et2[:,1][mask22],s=ds,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    
    ax.scatter(qp[mask12],bvals[mask12]/bmax,c=SC.et1[:,0][mask12],s=10,alpha=0,hatch='/')
    ax.scatter(qp[mask22],bvals[mask22]/bmax,s=10,alpha=0,hatch='/')

#We also make a legend handle
ghand = plt.Rectangle((0, 0), 1, 1, fc="darkgreen",label=r'$e_{i,crit}<\tilde{e}_i<1$')
hhand1 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',label='$\mathrm{Orbit\ crossing\ A}$')
hhand2 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',hatch='/',label='$\mathrm{Orbit\ crossing\ B}$')
#
#ax.set_yticks(np.arange(-ymax,ymax+0.1*ymax,0.1*ymax))

ax.set_xlabel('$q_p$')
#ax.set_ylabel(r'$b\ [R_\mathrm{Hill,m}]$')
ax.set_ylabel(r'$b\ [b_\mathrm{max}]$')

#We plot bmax versus Mstar
#bmplot, = ax.plot(qvals,bmvals,'k--',alpha=0.75,label='$\pm b_\mathrm{max}$')
#ax.plot(qvals,-bmvals,'k--',alpha=0.75,label='$b_\mathrm{max}$')

ax.set_xlim(qvals.min(),qvals.max())
ymax = 1
ax.set_ylim(-ymax,ymax)
ax.set_xscale('log')

cbar = fig.colorbar(scim,ax=ax,ticks=np.arange(0.8,1+0.05,0.05))
cbar.ax.set_ylabel(r'$\tilde{e}$')

cbox = cbar.ax.get_position() # get the original position 
cpos = [cbox.x0 + 0.05, cbox.y0,  cbox.width, cbox.height] 
cbar.ax.set_position(cpos) # set a new position

box = ax.get_position() # get the original position 
pos = [box.x0 + 0.05, box.y0,  box.width, box.height] 
ax.set_position(pos) # set a new position

#We add the current date
date = datetime.datetime.now()
datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
        
#fig.text(0.87,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
#fig.text(0.87,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)

#We also add a table

celldata  = [[a1,e1,'$301-m_2$'],[a2,e2,'$m_2\in[1,300]$']]
tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']

table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
          loc='top',cellLoc='center')

table.set_fontsize(11.8) 
table.scale(1, 1.44)

yticks = ax.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax.legend(handles=[ghand,hhand1,hhand2],prop={'size':13})

plt.savefig('zucc_qp_plot_{0}_{1}_{2}.png'.format(date.day,date.month,date.year),dpi=300)

os.chdir(os.getcwd()+'/../../Results/Zucc plots')
plt.savefig('zucc_qp_plot_{0}_{1}_{2}.png'.format(date.day,date.month,date.year),dpi=300)
#os.chdir(os.getcwd()+'/../../Report/Figures')
#plt.savefig('zucc_qp_plot_{0}_{1}_{2}.png'.format(date.day,date.month,date.year),dpi=300)
plt.close('all')