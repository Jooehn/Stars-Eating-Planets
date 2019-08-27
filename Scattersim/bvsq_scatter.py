"""
Created on Tue Apr 23 16:10:22 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
from matplotlib import colors
from scattersim import Scatter

plt.rcParams['font.size']= 16
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['hatch.linewidth'] = 2

rjtoau = 1/2150

a1, a2 = 30.0,30.1
e1, e2 = 0.2,0.0

#We set up the class object with orbits corresponding to the given parameters
#We can also choose to misalign the semi-major axes by an angle theta given
#in radians

theta = 0
Rstar = 1/215
Mstar = 1

#We set up a set of q values which we will investigate by setting up the values
#of our planets for each iteration.

earthtosunm = 1/332946

m2vals1 = np.logspace(np.log10(1),np.log10(300),200)
m1vals1 = (301-m2vals1)

m1vals = np.concatenate([m1vals1,m2vals1])*earthtosunm
m2vals = np.concatenate([m2vals1,m1vals1])*earthtosunm

qvals = m2vals/m1vals

#We also define a set of bvals for which we will evaluate for each q

bmax = 0.01
bvals = np.linspace(-bmax,bmax,1000)

#Set up a few empty containers

devals1 = np.zeros(len(qvals))
devals2 = np.zeros(len(qvals))

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

fig, ax = plt.subplots(figsize=(10,6))
fig2, ax2 = plt.subplots(figsize=(10,6)) 

for i in range(len(qvals)):
    
    #For each mass combination we perform a scattering and plot the result
    
    m1 = m1vals[i]
    m2 = m2vals[i]
    
    p1data = np.array([a1,e1,m1])
    p2data = np.array([a2,e2,m2])    
    
    SC = Scatter(p1data,p2data,Mstar,Rstar,theta=theta)
    
    #We perform the scatterings
    SC.scatter(b=bvals)
    
    #Next we find which planets have critically interacted with the host star
    #as well as which ones have collided
    
    mask11 = SC.scoll[:,0,0] & (SC.et1[:,0]<1)
    mask12 = SC.scoll[:,0,1] & (SC.et1[:,1]<1)
    
    mask21 = SC.scoll[:,1,0] & (SC.et2[:,0]<1)
    mask22 = SC.scoll[:,1,1] & (SC.et2[:,1]<1)
   
    col1 = np.asarray(['tab:green']*np.size(SC.b))
    col2 = np.asarray(['tab:green']*np.size(SC.b))

    #We also check for which cases we get a P-P merger
    col1[SC.dcrit>=SC.dmin] = 'tab:red'
    col2[SC.dcrit>=SC.dmin] = 'tab:red'
    
    #Further, we check if the total change of the eccentricity is small
    
    devals1[i] = abs(SC.et1[0]-e1).sum()
    devals2[i] = abs(SC.et2[0]-e2).sum()
    
#    if np.allclose(abs(SC.et1-e1),0):
#        devals1 = 1
#    if np.allclose(abs(SC.et2-e2),0):
#        devals2 = 1
    
    qp = np.asarray([qvals[i]]*len(bvals))

    ax.scatter(qp[mask11],bvals[mask11]/rjtoau,c=col1[mask11],s=3)
    ax.scatter(qp[mask12],bvals[mask12]/rjtoau,c=col1[mask12],s=3)
    ax.scatter(qp[mask12],bvals[mask12]/rjtoau,c=col1[mask11],s=7,alpha=0,hatch='/')
    ax.scatter(qp[mask21],bvals[mask21]/rjtoau,c=col2[mask21],s=3)
    ax.scatter(qp[mask22],bvals[mask22]/rjtoau,c=col2[mask22],s=3)
    ax.scatter(qp[mask21],bvals[mask22]/rjtoau,c=col2[mask22],s=7,alpha=0,hatch='/')

#    ax2.scatter(qp[mask11],SC.et1[:,0][mask11],c=col1[mask11],s=3,alpha=0.5)
#    ax2.scatter(qp[mask11],SC.et1[:,0][mask11],c=col1[mask11],s=7,alpha=0,hatch='/')
#    ax2.scatter(qp[mask12],SC.et1[:,1][mask12],c=col1[mask12],s=3,alpha=0.5)
#    ax2.scatter(qp[mask21],SC.et2[:,0][mask21],c=col2[mask21],s=3,alpha=0.5)
#    ax2.scatter(qp[mask21],SC.et2[:,0][mask21],c=col2[mask21],s=7,alpha=0,hatch='/')
#    ax2.scatter(qp[mask22],SC.et2[:,1][mask22],c=col2[mask22],s=3,alpha=0.5)

#We also make a legend handle
ghand = plt.Rectangle((0, 0), 1, 1, fc="tab:green",label=r'$e_{i,crit}<\tilde{e}_i<1$')
rhand = plt.Rectangle((0, 0), 1, 1, fc="tab:red",label=r'$d_{min}\leq d_{crit}$')
hhand1 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',label='$\mathrm{Orbit\ crossing\ A}$')
hhand2 = plt.Rectangle((0, 0), 1, 1, fc="None",ec='k',hatch='/',label='$\mathrm{Orbit\ crossing\ B}$')

ax.set_xlabel('$q_p$')
ax.set_ylabel('$b\ [R_J]$')

ax.set_xscale('log')
ax.set_xlim(qvals.min(),qvals.max())
ax.set_ylim(-bmax/rjtoau,bmax/rjtoau)

ax2.set_xlabel('$q_p$')
ax2.set_ylabel(r'$\tilde{e_i}$')

ax2.set_xscale('log')
ax2.set_xlim(qvals.min(),qvals.max())
ax2.set_ylim(0,1.1)

#We add the current date

date = datetime.datetime.now()
        
datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
        
fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
fig2.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)

#We also add a table

celldata  = [[a1,e1,'$m_1\in[1,300]$'],[a2,e2,'$301-m_1$']]
tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']

table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
          loc='top',cellLoc='center')

table.set_fontsize(10)

table.scale(1, 1.2)

yticks = ax.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

table2 = ax2.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
          loc='top',cellLoc='center')

table2.set_fontsize(10)

table2.scale(1, 1.2)

yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax.legend(handles=[ghand,rhand,hhand1,hhand2],prop={'size':13})
#plt.tight_layout()