"""
Created on Fri Sep 20 15:59:45 2019

@author: John Wimarsson
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from scattersim import Scatter
from plotfuncs import *

plt.close('all')

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
Rstar = 1/215
Mstar = 1
metoms = 1/332946
rjtoau = 1/2150

a1     = 10
e1, e2 = 0.0,0.9
m1, m2 = 300*metoms, metoms

#We set up a set of a values which we will investigate by setting up the values
#of our planets for each iteration.

avals = np.logspace(-1,2,500)

#We also define a set of bvals for which we will evaluate for each q

bmax = 0.01
bvals = np.linspace(-bmax,bmax,1000)

#Set up a few empty containers

devals1 = np.zeros(len(avals))
devals2 = np.zeros(len(avals))

bminarr11 = np.zeros(len(avals))
bmaxarr11 = np.zeros(len(avals))

bminarr12 = np.zeros(len(avals))
bmaxarr12 = np.zeros(len(avals))

bminarr21 = np.zeros(len(avals))
bmaxarr21 = np.zeros(len(avals))

bminarr22 = np.zeros(len(avals))
bmaxarr22 = np.zeros(len(avals))

amask11   = np.full(len(avals),False)
amask12   = np.full(len(avals),False)
amask21   = np.full(len(avals),False)
amask22   = np.full(len(avals),False)

#We then loop through all our q-values and save the points for which the upper
#and lower points in b for both of our orbital collision points.

#fig, ax = plt.subplots(figsize=(10,6))
fig, ax = plt.subplots(figsize=(10,6))

for i in range(len(avals)):
    
    #For each mass combination we perform a scattering and plot the result
    
    a2 = avals[i]
    
    p1data = np.array([a1,e1,m1])
    p2data = np.array([a2,e2,m2])    
    
    try:
        SC = Scatter(p1data,p2data,Mstar,theta=theta)
    except Exception:
        continue
    
    #We define a set of bvals for which we will evaluate for each qstar
    bmax  = SC.find_bmax(fac=0.01,step=0.001)
    bvals = np.linspace(-bmax,bmax,int(1e4)) 
    
    #Next we find which planets have critically interacted with the host star
    #as well as which ones have collided
    
    mask11 = SC.scoll[:,0,0]
    mask12 = SC.scoll[:,0,1]
    
    mask21 = SC.scoll[:,1,0]
    mask22 = SC.scoll[:,1,1]
   
    col1 = np.asarray(["tab:red"]*np.size(SC.b))
    col2 = np.asarray(["tab:red"]*np.size(SC.b))

    #We also check for which cases we get a P-P merger
    col1[~SC.merger[:,0]] = 'None'
    col2[~SC.merger[:,1]] = 'None'
    
    #Further, we check if the total change of the eccentricity is small
    
    devals1[i] = abs(SC.et1[0]-e1).sum()
    devals2[i] = abs(SC.et2[0]-e2).sum()
    
#    if np.allclose(abs(SC.et1-e1),0):
#        devals1 = 1
#    if np.allclose(abs(SC.et2-e2),0):
#        devals2 = 1
    
    ap = np.asarray([avals[i]]*len(bvals))

    scim = ax.scatter(ap[mask11],bvals[mask11]/bmax,c=SC.et1[:,0][mask11],s=5,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(ap[mask12],bvals[mask12]/bmax,c=SC.et1[:,1][mask12],s=5,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(ap[mask21],bvals[mask21]/bmax,c=SC.et2[:,0][mask21],s=5,\
                vmin = 0.8, vmax = 1,cmap='Greens')
    ax.scatter(ap[mask22],bvals[mask22]/bmax,c=SC.et2[:,1][mask22],s=5,\
                vmin = 0.8, vmax = 1,cmap='Greens')
   
#    ax.scatter(ap[mask11],bvals[mask11]/bmax,c=col1[mask11],s=3)
#    ax.scatter(ap[mask12],bvals[mask12]/bmax,c=col1[mask12],s=3)
#    ax.scatter(ap[mask21],bvals[mask21]/bmax,c=col2[mask21],s=3)
#    ax.scatter(ap[mask22],bvals[mask22]/bmax,c=col2[mask22],s=3)
    
    ax.scatter(ap[mask12],bvals[mask12]/bmax,c=SC.et1[:,0][mask12],s=10,alpha=0,hatch='/')
    ax.scatter(ap[mask22],bvals[mask22]/bmax,s=10,alpha=0,hatch='/')

#We also make a legend handle
ghand = plt.Rectangle((0, 0), 1, 1, fc="tab:green",label=r'$e_{i,crit}<\tilde{e}_i<1$')
rhand = plt.Rectangle((0, 0), 1, 1, fc="tab:red",label=r'$d_{min}\leq d_{crit}$')
hhand1 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',label='$\mathrm{Orbit\ crossing\ A}$')
hhand2 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',hatch='/',label='$\mathrm{Orbit\ crossing\ B}$')

ax.set_xscale('log')
ax.set_xlim(avals.min(),avals.max())
ax.set_ylim(-1,1)

ax.set_xlabel('$a_2\ \mathrm{[AU]}$')
ax.set_ylabel(r'$b\ [\mathrm{R_J}]$')

#divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(scim,ax=ax)
cbar.ax.set_ylabel(r'$\tilde{e}$')

#We add the current date

#date = datetime.datetime.now()
#        
#datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
#        
#fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
#fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)

#We also add a table

celldata  = [[a1,e1,int(m1/metoms)],['$a_2\in[0.1,100]$',e2,int(m2/metoms)]]
tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']

table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
          loc='top',cellLoc='center')

table.set_fontsize(10)

table.scale(1, 1.2)

yticks = ax.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

#table2 = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
#          loc='top',cellLoc='center')
#
#table2.set_fontsize(10)
#
#table2.scale(1, 1.2)
#
#yticks = ax.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

ax.legend(handles=[ghand,rhand,hhand1,hhand2],prop={'size':13})

plt.savefig('zucc_plot.png',dpi=300)
#plt.tight_layout()