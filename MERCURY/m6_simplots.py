#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:51:15 2019

@author: John Wimarsson

Script that plots the outcomes for the different mass cases for our simulations
"""

from m6_output import *
import os

def plot_scp_fraction(dirs):
    """Function that plots the outcome fractons for each case in the 2X+3J runs.
    Also plots the fraction of consumed planets for a given mass of the innermost
    planets."""
    Ncases = len(dirs)
    
    outcomes = np.zeros((Ncases,4))
    massdepo = np.zeros(Ncases)
    alpha    = np.zeros(Ncases)
    
    pwd = os.getcwd()
    
    for i in range(Ncases):
        
        os.chdir(pwd+'/'+dirs[i])
 
        alpha = get_masses[0]/metoms
       
        siminfo = np.loadtxt('siminfo.txt',skiprows=1)
        
        outcomes[i,0] = np.sum(siminfo[:,1]) #Survivors
        outcomes[i,1] = np.sum(siminfo[:,4]) #Mergers
        outcomes[i,2] = np.sum(siminfo[:,3]) #Ejections
        outcomes[i,3] = np.sum(siminfo[:,2]) #Star-planet collisions
        
        #We now want to read in the masses deposited into the system
        m6d = m6_load_data(ce_data=False)
    
        m6a = m6_analysis(m6d)
        
        pmass = m6a.fin_phases[:,:,7]
        
        sysid = m6a.sclist[0]
        plaid = m6a.sclist[1]
                
        massdepo[i] += pmass[sysid,plaid].sum()/m6a.metoms
        
        del m6d,m6a
        
    Nplanet = np.sum(siminfo[0][1:5])
    Nrealis = len(siminfo)

    foutcomes = outcomes/(Nplanet*Nrealis)

    os.chdir(pwd)
    
    #First we plot the fractions of all outcomes in the bar plot
    fig, ax = plt.subplots(figsize=(12,6))
    
    #We plot each fraction in order: survivors, mergers, ejections and SPC
    
    yoffset = np.zeros(Ncases)
    clist   = ['tab:blue','tab:grey','tab:orange','tab:green']
    for i in range(len(foutcomes)):
        
        plt.bar(alpha,foutcomes[:,i],bottom=yoffset,color=clist[i])
        
        yoffset += foutcomes[:,i]
        
    ax.set_xlabel(r'$\alpha\ [\mathrm{M}_\oplus]$')
    ax.set_ylabel('$f_\mathrm{outcome}$')
    
    ax.set_ylim(-0.1,1.1)
    
    #################### Star-planet collision fraction #######################
    
    fig2, ax2 = plt.subplots(figsize = (12,6))
    
    ax2.plot(alpha,foutcomes[:,3],label='$f_\mathrm{useful}$')
    
    ax2.set_xlabel(r'$\alpha\ [\mathrm{M}_\oplus]$')
    ax2.set_ylabel('$f_\mathrm{outcome}$')
    
    ax2n = ax2.twinx()
    
    ax2n.plot(alpha,massdepo,ls='--',label=r'$\alpha$')
    ax2n.set_ylim(0,1e4)
    ax2n.set_ylabel(r'$\mathrm{Mass\ deposited\ [M]}_\oplus$')
    
######## Call functions ########
    
dirs = ['1Me','30Me','100Me','300Me']    
    
plot_scp_fraction()