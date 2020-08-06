#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:25:04 2020

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import _pickle as cPickle
from matplotlib import colors
from scattersim import Scatter
from plotfuncs import *

rjtoau = 1/2150
metoms = 1/332946
Mstar = 1

def m6_load_data(filename = None):
    try:
        data = cPickle.load(open(filename,'rb'))
        
    except FileNotFoundError:
        
        print('No data to be found. You have yet to run a simulation or the file\
               name is wrong.')
    
    return data

mxvals = [1,3,10,30,100,300]
pmass = False

cwd = os.getcwd()

for i in range(len(mxvals)):
    
    mxval = mxvals[i]
    plt.close('all')
    os.chdir('/home/jooehn/Documents/Uni/MSc/Data/MERCURY/2X+3J/CE data/')
    
    fced,lanvals = m6_load_data('2X+3J_{}Me_firstce.pk1'.format(mxval))
    
    os.chdir(cwd)
    
    #We set up some parameters
    
    K, N = lanvals.shape
    
    mc_axes = None
    mc_figs = None
    
    names = ['X1','X2','J1','J2','J3']
    clist = ['m','b','g','r','orange']
    mvals = [mxval*metoms,mxval*metoms,300*metoms,300*metoms,300*metoms]
    
    for k in range(K):
        
        fcen = fced[k]
        
        ilist = list(range(N))
        for i in ilist:
            
            if np.all(pd.isnull(fcen[i])):
                    continue
            duplist = []
            for j in ilist:
                if np.all(pd.isnull(fcen[j])):
                    continue
                dup = (fcen[i][0] == fcen[j][0]) & (fcen[i][2] == fcen[j][2])
                duplist.append(dup)
            if sum(duplist) > 1:
                ilist.remove(i)
                
        fcen = [fcen[i] for i in ilist]
        
        print('Currently on system: ',k)
        
        for n in range(len(fcen)):
            
            fce = fcen[n]
            
            if np.all(pd.isnull(fce)):
                continue
            
            plaid = ilist[n]
            tarid = names.index(fce[1])
            
            a1 = fce[3]
            e1 = fce[4]
            a2 = fce[6]
            e2 = fce[7]
            m1, m2 = mvals[plaid], mvals[tarid]
            
            if any([i == '*********' for i in [a1,e1,a2,e2]]):
                continue
            
            p1data = np.array([a1,e1,m1],dtype=float)
            p2data = np.array([a2,e2,m2],dtype=float)
    
            theta = lanvals[k][plaid]
    
            try:
                SC = Scatter(p1data,p2data,Mstar,theta=theta)
            except ValueError:
                continue
            SC._mc_axes = mc_axes
            SC._mc_figs = mc_figs
            
            SC.collfinder(1,disp=False,col1=clist[plaid],col2=clist[tarid],pmass=pmass)
            
            mc_figs = SC._mc_figs
            mc_axes = SC._mc_axes
            
    _,_,_,ax3,ax4,ax5 = SC._mc_axes
    ax3.set_title('$\mathrm{2X+3J\ in\ 2D\ for}\ M_\mathrm{x} = '+str(mxval)+'\ \mathrm{M}_\oplus$')
    ax4.set_title('$\mathrm{2X+3J\ in\ 2D\ for}\ M_\mathrm{x} = '+str(mxval)+'\ \mathrm{M}_\oplus$')
    ax5.set_title('$\mathrm{2X+3J\ in\ 2D\ for}\ M_\mathrm{x} = '+str(mxval)+'\ \mathrm{M}_\oplus$')
    
    fig, fig2, fig3, fig4, fig5 = SC._mc_figs
    
    os.chdir(cwd+'/../../Results/Scattersim MC')
    fig3.savefig('2X+3J_'+str(mxval)+'Me_rmin_vs_Nce.pdf')
    fig4.savefig('2X+3J_'+str(mxval)+'Me_e1_vs_e2.pdf')
    fig5.savefig('2X+3J_'+str(mxval)+'Me_vinf_vs_dmin.pdf')
    os.chdir(cwd)