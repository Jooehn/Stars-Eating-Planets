#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:29:36 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from plotfuncs import *
from astropy.io import ascii

pwd = os.getcwd()
os.chdir(pwd+'/../../Data')

#We load our exoplanet data
pdata = ascii.read('NASA_planet_data.csv')

#and set up our constants

metoms = 1/332946
mjtoms = 300*metoms
mstogr = 1.99e33
metomj = 1/300
rjtoau = 1/2150
retoau = rjtoau/11
retorj = 1/11
rstoau = 1/215
rjtors = rjtoau/rstoau
autocm = 14959790000000

#Furtermore, we create arrays with the solar system planet data normalised to
#Jupiter values

m_ss    = np.array([0.330,4.87,5.97,0.642,1898,568,86.8,102])/5.97
r_ss    = np.array([4879,12104,12756,6792,142984,120536,51118,49528])/12756
rho_ss  = np.array([5.427,5.243,5.514,3.933,1.326,0.687,1.271,1.638])
c_ss    = ['magenta','y','g','r','darkorange','purple','c','b']
name_ss = ['Mercury','Venus','Earth','Mars','Jupiter',\
           'Saturn','Uranus','Neptune']

os.chdir(pwd)

plt.close('all')

pmass = pdata['pl_bmassj']/metomj
prad  = pdata['pl_radj']/retorj

mvals_rocky = np.logspace(-1.3,np.log10(2.62),100)
mvals       = np.logspace(np.log10(2.62),3.5,100)

def calc_R(mass):
    rvals = 10**(0.087+0.141*np.log10(mass*metomj)-0.171*np.log10(mass*metomj)**2)
    
    return rvals/retorj

def calc_R_rocky(mass):
    CMF = 0.33 #Core mass fraction of Earth
    
    rvals = (1.07-0.21*CMF)*(mass)**(1/3.7)
    
    return rvals

rvals = calc_R(mvals)
rvals_rocky = calc_R_rocky(mvals_rocky)

fig, ax = plt.subplots(figsize=(8,7))

ax.plot(mvals,rvals,'-',zorder=1,color='red')
ax.plot(mvals_rocky,rvals_rocky,ls = '--',zorder=1,color='blue')
ax.scatter(pmass,prad,c='k',s=3)
ax.scatter(m_ss,r_ss,c=c_ss,s=30,zorder=2)

ax.set_xlabel(r'$M_p\ [\mathrm{M}_\oplus]$')
ax.set_ylabel(r'$R_p\ [\mathrm{R}_\oplus]$')

#ax.set_xlim(1e-4,1e2)
ax.set_ylim(0.1,50)

ax.set_xscale('log')
ax.set_yscale('log')

########## Density estimation ##########

def calc_rho(mass,R):
    
    Vvals = 4*np.pi*(R*retoau*autocm)**3/3

    rho = mass*metoms*mstogr/Vvals
    
    return rho

rho = calc_rho(mvals,rvals)
rho_rocky = calc_rho(mvals_rocky,rvals_rocky)

fig2, ax2 = plt.subplots(figsize=(8,6))

TD12,  = ax2.plot(mvals,rho, color='red',zorder=1,label='Tremaine\ \& Dong\ (2012)')
ZSJ16, = ax2.plot(mvals_rocky,rho_rocky,ls = '--',color='blue',zorder=1,label='Zeng, Sasselov \& Jacobsen (2016)')

prho = pdata['pl_dens']
prhoc = prho[~prho.mask]
pmassc = pmass[~prho.mask]

ax2.scatter(pmassc,prhoc,c='k',s=3)
ax2.scatter(m_ss,rho_ss,c=c_ss,s=30,zorder=2)

ax2.set_xlabel(r'$M_p\ [\mathrm{M}_\oplus]$')
ax2.set_ylabel(r'$\bar{\rho}_p\ [\mathrm{g\ cm}^{-3}]$')

ax2.set_ylim(0.01,100)

ax2.set_xscale('log')
ax2.set_yscale('log')

ax.axvline(2.62,ls='--',color='tab:grey')
ax2.axvline(2.62,ls='--',color='tab:grey')

#We set up a legend

exoh = ax2.scatter([],[],c='k',s=3,label='Confirmed exoplanets')
hlist = [TD12,ZSJ16,exoh]

for i in range(len(c_ss)):
    handle = ax2.scatter([],[],c=c_ss[i],s=30,zorder=2,label=name_ss[i])
    hlist.append(handle)

#box = ax.get_position()
#ax.set_position([box.x0-0.065, box.y0, box.width * 0.97, box.height])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.86])
#fig.subplots_adjust(wspace=0.1)

# Put a legend to the right of the current axis    
ax.legend(handles=hlist,loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.25),prop={'size':12})

#ax1.legend(prop={'size':12})

os.chdir(pwd+'/../../Report/Figures/Exoplanets')

fig.savefig('m_r_relation.png',dpi=100)
fig2.savefig('m_rho_relation.png',dpi=100)