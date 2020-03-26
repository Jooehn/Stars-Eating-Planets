#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:44:41 2019

@author: John Wimarsson

Script that plots properties for exoplanets and their host stars
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from plotfuncs import *
from astropy.io import ascii

pwd = os.getcwd()
os.chdir(pwd+'/../../Data')

#We load our exoplanet data
pdata = ascii.read('NASA_planet_data2.csv')

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

os.chdir(pwd)

#Furtermore, we create arrays with the solar system planet data normalised to
#Jupiter values

m_ss    = np.array([0.330,4.87,5.97,0.642,1898,568,86.8,102])/5.97
r_ss    = np.array([4879,12104,12756,6792,142984,120536,51118,49528])/12756
rho_ss  = np.array([5.427,5.243,5.514,3.933,1.326,0.687,1.271,1.638])
c_ss    = ['magenta','y','g','r','darkorange','purple','c','b']
name_ss = ['Mercury','Venus','Earth','Mars','Jupiter',\
           'Saturn','Uranus','Neptune']

def est_L_zams(m):
    """We use the analytical prescription from Tout et al. (1996) to estimate the
    ZAMS luminosities for our HR-D"""
    
    #We load in the coefficients for the constants
    C = np.loadtxt('zams_coeff.txt',skiprows=1,delimiter=',',usecols=[1,2,3,4,5])[:7]
    
    #We assume Solar metallicity and compute eq. (3)
    Z = 0
#    theta = np.sum(C,axis=1)
    alpha = C[:,0]+C[:,1]*np.log10(Z/Z_s)+C[:,2]*np.log10(Z/Z_s)**2+\
            C[:,3]*np.log10(Z/Z_s)**3+C[:,4]*np.log10(Z/Z_s)**4
    
    #We now use equation (1) from Tout et al. (1996) to estimate L
    
    Lzams = (alpha[0]*m**5.5 + alpha[1]*m**11)/(alpha[2]+m**3+alpha[3]*m**5+\
                alpha[4]*m**7+alpha[5]*m**8+alpha[6]*m**9.5)
    
    return Lzams
    
def est_R_zams(m,Z=0.02):
    """We use the analytical prescription from Tout et al. (1996) to estimate the
    ZAMS luminosities for our HR-D"""
    
    #We load in the coefficients for the constants
    C = np.loadtxt('zams_coeff.txt',skiprows=1,delimiter=',',usecols=[1,2,3,4,5])[7:]
    
    #We assume Solar metallicity and compute eq. (4)
    Z_s = 0.02

    theta = C[:,0]+C[:,1]*np.log10(Z/Z_s)+C[:,2]*np.log10(Z/Z_s)**2+\
            C[:,3]*np.log10(Z/Z_s)**3+C[:,4]*np.log10(Z/Z_s)**4
    #We now use equation (2) from Tout et al. (1996) to estimate R for ZAMS
    
    Rzams = (theta[0]*m**2.5 + theta[1]*m**6.5+theta[2]*m**11+theta[3]*m**19+theta[4]*m**19.5)/\
            (theta[5]+theta[6]*m**2+theta[7]*m**8.5+m**18.5+theta[8]*m**19.5)
    
    return Rzams

def plot_HRD(sample):
    
    """Function that given a sample and its G and BP_RP magnitudes provides an H-R diagram with the density of stars
    provided using a colormap. The density is computed using a Gaussian KDE"""
    
#    parallax = sample['gaia_plx']
#    plxerr   = sample['gaia_plxerr1']
    
#    plxmas = (plxerr/parallax)<0.1
    
#    sample_cut = sample[plxmas]
    
    Teff     = sample['st_teff']
    Lstar    = sample['st_lum']
    Zvals    = sample['st_metfe']
    
    mask = ~Teff.mask & ~Lstar.mask & ~Zvals.mask
    
    Teff     = Teff[mask]
    Lstar    = Lstar[mask]
    Zvals    = Zvals[mask]
    
    xmin = 2.5e3
    xmax = 1.1e4
    ymin = -4
    ymax = 2
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    
    ax.set_title('$\mathrm{HR-Diagram\ of\ planet\ hosts}$',size='large')
    ax.set_xlabel('$T_\mathrm{eff}$')
    ax.set_ylabel(r'$L_\star\ [\mathrm{L}_\odot]$')
    
    
#    cb.set_label('$\mathrm{Number\ density\ of\ stars}$',size='large')
    
    plt.show()
    
    return

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xscale('log')
    stars = ax.scatter(Teff,Lstar,c=Zvals,edgecolor='',s=10,cmap=plt.cm.get_cmap('viridis'))#,norm=LogNorm())
    ax.invert_xaxis()
def plot_Rstar_vs_Mstar(sample):
    """We plot the radius as a function of the mass for our stars along
    with an analytically derived ZAMS radius to see if exoplanets are usually
    have host stars that are on the main sequence."""
    
    Mstar     = sample['st_mass']
    Mstar_err = 0.5*(sample['st_masserr1']+abs(sample['st_masserr2']))
    Rstar     = sample['st_rad']
    Rstar_err = 0.5*(sample['st_raderr1']+abs(sample['st_raderr2']))
    Zvals     = sample['st_metfe']
#    Zvals    = sample['st_metfe']
    
    sigma_M = Mstar_err/Mstar
    sigma_R = Rstar_err/Rstar
    
    mask = ~Mstar.mask & ~Rstar.mask & ~Zvals.mask & (sigma_M<0.1) & (sigma_R<0.1)
    
    Mstar    = Mstar[mask]
    Rstar    = Rstar[mask]
    Zvals    = Zvals[mask]
#    Zvals    = Zvals[mask]
    
    #We estimate the ZAMS radius for a set of stellar masses
    
    Mstar_zams = np.logspace(-2,2,100)
    Rstar_zams = est_R_zams(Mstar_zams)
    
    #We also want to colour the scatter points by their stellar age. If they do
    #not have an age we set the value to nan
    
    Zvals[Zvals.mask] = np.nan
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    
    Mmin = 0.1
    Mmax = 11
    Rmin = 0.1
    Rmax = 11
    
    ax.set_title('$\mathrm{MR-relation\ for\ planet\ hosts}$',size='large')
    ax.set_xlabel(r'$M_\star\ [\mathrm{M}_\odot]$')
    ax.set_ylabel(r'$R_\star\ [\mathrm{R}_\odot]$')
    
    ax.set_xlim(Mmin,Mmax)
    ax.set_ylim(Rmin,Rmax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    stars = ax.scatter(Mstar,Rstar,c=Zvals,edgecolor='k',lw=0.3,s=10,cmap=plt.cm.get_cmap('viridis'),\
                       label='Confirmed exoplanet hosts',vmin=-0.5,vmax=0.5)
    cbar = plt.colorbar(stars)
    cbar.set_label('[Fe/H]')
#    stars = ax.scatter(Mstar,Rstar,s=3,label='Exoplanet host stars')
    mrzams = ax.plot(Mstar_zams,Rstar_zams,'k--',label = 'ZAMS MR-relation (Tout+96)',alpha=0.75)
    ax.legend(loc='lower right',prop={'size':12})
    ax.text(4,1.7,'$Z=Z_\odot$')
    
    os.chdir(pwd+'/../../Results')
    fig.savefig('MR_relation_ZAMS.png',dpi=100)

plt.close('all')
#plot_HRD(pdata)
plot_Rstar_vs_Mstar(pdata)