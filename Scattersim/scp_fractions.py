#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:36:41 2019

@author: John Wimarsson

A script that investigates for which combinations of semi-major axis and 
eccentricity that can lead to at least one star-planet collision and how
the fraction of combinations in our parameter space that lead to star-planet
collision varies with mass of the secondary component.
"""

from scattersim import *

rjtoau = 1/2150
metoms = 1/332946

def get_usefulness(a1=1,m2=metoms,Mstar=1):
    """This function evaluates the usefulness for each point in the two-dimensional
    parameter space for a2 and e2 and returns the fraction for each outcome in an
    array."""

    no_cross = [[],[]]
    
    #fvals contains all the fractions for each outcome, index 0 is SPC, index 1 is
    #ejection and index 2 is merger
    
    fvals = np.zeros((3,N_a,N_e))
    
    #We loop through our parameter space and perform a scattering for each system
    #where a1, e1 are kept static and a2, e2 vary
    
    for i in range(N_a):
        
        a2 = avals[i]
        
        for j in range(N_e):
            
            e2 = evals[j]
            
            p1data = np.array([a1,e1,m1])
            p2data = np.array([a2,e2,m2])
            
            #We set up the two orbits, if they do not cross we log the indices of
            #the corresponding orbital parameters
            try:
                SC = Scatter(p1data,p2data,Mstar,theta=theta)
            except ValueError as err:
                if err.args[0] not in ['The orbits do not cross']:
                    print(err.args,'for a2 = {0} and e2 = {1}'.format(a2,e2))
                no_cross[0].append(i)
                no_cross[1].append(j)
                continue
            
            #We find an appropriate range of b-values for the given configuration
            bmax = SC.find_bmax()
            
            bvals = np.linspace(-bmax,bmax,N_scatter)
            
            #Next we carry out the scatterings
            SC.scatter(b=bvals)
            
            #We save the fraction for each outcome. In order to normalize we need
            #to divide by N_scatter times two, as we have two crossing points.
            #We do have two planets per case, but we cannot have both planets
            #hit the star due to conservation of energy and angular momentum
            fvals[0,i,j] = SC.scoll.sum()/(N_scatter*2)
            fvals[1,i,j] = SC.eject.sum()/(N_scatter*2)
            fvals[2,i,j] = SC.merger.sum()/(N_scatter*2)
        
    return fvals, no_cross

def usefulness_vs_qp(Mstar=1):
    """Plots the usefulness of four different a1-value orbits given (a2,e2) space
    as compared to each other. Normalised by the orbit with maximum usefulness."""
    qp = np.logspace(-3,0,50)
    
    a1vals = [0.3,1,3,8]
    
    ftot = np.zeros((len(a1vals),len(qp)))
    
    for i in range(len(a1vals)):
    
        a1 = a1vals[i]
        
        for j in range(len(qp)):
            if not j%10:    
                print('Currently on iteration: '+str(i)+'.'+str(j))
            fvals,_ = get_usefulness(a1,m2=qp[j]*m1,Mstar=Mstar)
            
            ftot[i,j] = fvals[0].sum()
#            ftot[i,j] = fvals[0][np.nonzero(fvals[0])].mean()
#        
#    ftot[np.isnan(ftot)] = 0
    
    fnorm1 = ftot[0]/ftot.max()
    fnorm2 = ftot[1]/ftot.max()
    fnorm3 = ftot[2]/ftot.max()
    fnorm4 = ftot[3]/ftot.max()
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.plot(qp,fnorm1,'k--',label='$a_1 = 0.3\ \mathrm{AU}$')
    ax.plot(qp,fnorm2,'k-',label='$a_1 = 1\ \mathrm{AU}$')
    ax.plot(qp,fnorm3,'k.',label='$a_1 = 3\ \mathrm{AU}$')
    ax.plot(qp,fnorm4,'k-.',label='$a_1 = 8\ \mathrm{AU}$')
    
    ax.set_xlabel(r'$q_p$')
    ax.set_ylabel(r'$f_\mathrm{consumed,a_2}/\max{(f_\mathrm{consumed,a_2})}$')
#    ax.set_ylabel(r'$\langle f_{\mathrm{consumed},q_p} \rangle / \max\langle f_\mathrm{consumed,\alpha} \rangle$')
    ax.set_xlim(1.1e-3,1.1)
    ax.set_ylim(-0.1,1.1)
    ax.set_xscale('log')
        
    ax.legend(prop={'size':12})
    
    #We add the current date
    date = datetime.datetime.now()
            
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
            
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
    #We also add a table
    celldata  = [['$a_1=0.3,1,3,8$',e1,'{:.1f}'.format(m1/metoms)],['$a_2\in[0.1,10]$',\
                  '$e_2\in'+'[{0:.2f},{1:.2f}]'.format(evals.min(),evals.max())+'$',r'$q_p m_1$']]
    tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
    tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
    
    table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
              loc='top',cellLoc='center')
    
    table.set_fontsize(10)
    
    table.scale(1, 1.2)
    
def usefulness_vs_qstar(m2=metoms):
    """Plots the usefulness of four different a1-value orbits given (a2,e2) space
    as compared to each other for varying stellar mass values. Normalised by the 
    orbit with maximum usefulness."""
    
    Msvals = np.logspace(-1,1,50)
    
    a1vals = [0.3,1,3,8]
    
    ftot = np.zeros((len(a1vals),len(Msvals)))
    
    for i in range(len(a1vals)):
    
        a1 = a1vals[i]
        
        for j in range(len(Msvals)):
            if not j%10:    
                print('Currently on iteration: '+str(i)+'.'+str(j))
            fvals,_ = get_usefulness(a1,m2,Msvals[j])
            
            ftot[i,j] = fvals[0].sum()
#            ftot[i,j] = fvals[0][np.nonzero(fvals[0])].mean()
#        
#    ftot[np.isnan(ftot)] = 0
    
    fnorm1 = ftot[0]/ftot.max()
    fnorm2 = ftot[1]/ftot.max()
    fnorm3 = ftot[2]/ftot.max()
    fnorm4 = ftot[3]/ftot.max()
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(Msvals,fnorm1,'k--',label='$a_1 = 0.3\ \mathrm{AU}$')
    ax.plot(Msvals,fnorm2,'k-',label='$a_1 = 1\ \mathrm{AU}$')
    ax.plot(Msvals,fnorm3,'k.',label='$a_1 = 3\ \mathrm{AU}$')
    ax.plot(Msvals,fnorm4,'k-.',label='$a_1 = 8\ \mathrm{AU}$')
    
    ax.set_xlabel(r'$M_\star$')
    ax.set_ylabel(r'$f_\mathrm{consumed,a_1}/\max{(f_\mathrm{consumed,a_1})}$')
#    ax.set_ylabel(r'$\langle f_{\mathrm{consumed},q_p} \rangle / \max\langle f_\mathrm{consumed,\alpha} \rangle$')
    ax.set_xlim(0.1,10)
    ax.set_ylim(-0.1,1.1)
    ax.set_xscale('log')
        
    ax.legend(prop={'size':12})
    
    #We add the current date
    date = datetime.datetime.now()
            
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
            
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
    #We also add a table
    celldata  = [['$a_1=0.3,1,3,8$',e1,'{:.1f}'.format(m1/metoms)],['$a_2\in[0.1,10]$',\
                  '$e_2\in'+'[{0:.2f},{1:.2f}]'.format(evals.min(),evals.max())+'$',\
                  '{:.1f}'.format(m2/metoms)]]
    tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
    tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
    
    table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
              loc='top',cellLoc='center')
    
    table.set_fontsize(10)
    
    table.scale(1, 1.2)
    
def plot_outcome_fractions(a1=1,Mstar=1):
    """Iterates over a set of qp values and finds the total predicted fractions
    of each outcome, i.e. merger, ejection and star-planet collision and plots
    it in a graph with the fraction for each outcome as a function of qp"""
    
    qp = np.logspace(-3,0,50)
    
    ftot = np.zeros((4,len(qp)))
        
    for j in range(len(qp)):
        if not j%10:    
            print('Currently on iteration: '+str(j))
        fvals,_ = get_usefulness(a1,m2=qp[j]*m1,Mstar=Mstar)
        
        ftot[0,j] = fvals[0].sum()
        ftot[1,j] = fvals[1].sum()
        ftot[2,j] = fvals[2].sum()
        ftot[3,j] = (1-fvals.sum(axis=0)).sum()
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    zeros = np.zeros(len(qp))
    ones = np.ones(len(qp))
    
    #We normalise with respect to the total number of points in (a2,e2)-space
    #as the total fraction is 1*N_points
    
    fnorm = ftot/np.size(fvals[0])
    
    ax.fill_between(qp,zeros,fnorm[2],color='tab:grey',\
                    label='$\mathrm{Merger}$',alpha=0.6)
    ax.fill_between(qp,fnorm[2],fnorm[2]+fnorm[1],color='tab:orange',\
                    label='$\mathrm{Ejection}$',alpha=0.6)
    ax.fill_between(qp,fnorm[2]+fnorm[1],fnorm[2]+fnorm[1]+fnorm[0],color='tab:green',\
                    label='$\mathrm{Consumption}$',alpha=0.6)
    ax.fill_between(qp,fnorm[2]+fnorm[1]+fnorm[0],ones,color='w')
    
    ax.set_xlabel(r'$q_p$')
    ax.set_ylabel(r'$f_\mathrm{outcome}$')
    ax.set_xlim(1e-3,1)
    ax.set_ylim(0,0.25)
    ax.set_xscale('log')

    ax.set_yticks([0,0.05,0.1,0.15,0.2])
    
#    yticks = ax.yaxis.get_major_ticks()
#    yticks[-1].label1.set_visible(False)
        
    ax.legend(prop={'size':12},loc='upper right')
    
    #We add the current date
    date = datetime.datetime.now()
            
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
            
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
    #We also add a table
    celldata  = [[a1,e1,'{:.1f}'.format(m1/metoms)],['$a_2\in[0.1,10]$','$e_2\in'+\
                 '[{0:.2f},{1:.2f}]'.format(evals.min(),evals.max())+'$',\
                  r'$q_p m_1$']]
    tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
    tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
    
    table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
              loc='top',cellLoc='center')
    
    table.set_fontsize(10)
    
    table.scale(1, 1.2)
    
def plot_outcome_fractions_qstar(a1=1,m2=metoms):
    """Iterates over a set of qp values and finds the total predicted fractions
    of each outcome, i.e. merger, ejection and star-planet collision and plots
    it in a graph with the fraction for each outcome as a function of qp"""
    
    Msvals = np.logspace(-1,1,50)
    
    ftot = np.zeros((4,len(Msvals)))
        
    for j in range(len(Msvals)):
        if not j%10:    
            print('Currently on iteration: '+str(j))
        fvals,_ = get_usefulness(a1,m2,Mstar=Msvals[j])
        
        ftot[0,j] = fvals[0].sum()
        ftot[1,j] = fvals[1].sum()
        ftot[2,j] = fvals[2].sum()
        ftot[3,j] = (1-fvals.sum(axis=0)).sum()
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    zeros = np.zeros(len(Msvals))
    ones = np.ones(len(Msvals))
    
    #We normalise with respect to the total number of points in (a2,e2)-space
    #as the total fraction is 1*N_points
    
    fnorm = ftot/np.size(fvals[0])
    
    ax.fill_between(Msvals,zeros,fnorm[2],color='tab:grey',\
                    label='$\mathrm{Merger}$',alpha=0.6)
    ax.fill_between(Msvals,fnorm[2],fnorm[2]+fnorm[1],color='tab:orange',\
                    label='$\mathrm{Ejection}$',alpha=0.6)
    ax.fill_between(Msvals,fnorm[2]+fnorm[1],fnorm[2]+fnorm[1]+fnorm[0],color='tab:green',\
                    label='$\mathrm{Consumption}$',alpha=0.6)
    ax.fill_between(Msvals,fnorm[2]+fnorm[1]+fnorm[0],ones,color='w')
    
    ax.set_xlabel(r'$M_\star$')
    ax.set_ylabel(r'$f_\mathrm{outcome}$')
    ax.set_xlim(0.1,10)
    ax.set_ylim(0,0.25)
    ax.set_xscale('log')

    ax.set_yticks([0,0.05,0.1,0.15,0.2])
    
#    yticks = ax.yaxis.get_major_ticks()
#    yticks[-1].label1.set_visible(False)
        
    ax.legend(prop={'size':12},loc='upper right')
    
    #We add the current date
    date = datetime.datetime.now()
            
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
            
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
    #We also add a table
    celldata  = [[a1,e1,'{:.1f}'.format(m1/metoms)],['$a_2\in[0.1,10]$','$e_2\in'+\
                 '[{0:.2f},{1:.2f}]'.format(evals.min(),evals.max())+'$',\
                 '{:.1f}'.format(m2/metoms)]]
    tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
    tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
    
    table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
              loc='top',cellLoc='center')
    
    table.set_fontsize(10)
    
    table.scale(1, 1.2)
        
def plot_usefulness(a1=1,m2=metoms,Mstar=1):
    """Plots the normalised usefulness fraction as compared between various
    configurations with static a1 across the (a2,e2)-space."""

    fvals, nc = get_usefulness(a1,m2,Mstar=Mstar)
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    cmap = plt.cm.get_cmap('Blues')
    cmap.set_under('w')
    
    vmax  = 1.0
    vstep = 0.1*vmax
    
    contf = ax.contourf(avals,evals,fvals[0].T,cmap=cmap,vmin = 0,vmax = vmax,\
                        levels = np.arange(0,vmax+vstep,vstep),extend='neither')
    ax.contour(avals,evals,fvals[0].T,colors='k',vmin = 0,vmax = vmax,\
                        levels = np.arange(0,vmax+vstep,vstep),extend='neither')
    
    cbar = plt.colorbar(contf,ax=ax)
    cbar.set_label('$f_\mathrm{consumed}$')
    
    ax.set_xlabel('$a_2\ \mathrm{[AU]}$')
    ax.set_ylabel('$e_2$')
    ax.set_xlim(0.1,10)
    ax.set_ylim(0.7,0.99)
    ax.set_xscale('log')
    
    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    #ax.set_title('$\mathrm{Fraction\ of\ scatterings\ that\ are\ useful}$')
    
    #We add the current date
    date = datetime.datetime.now()
            
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
            
    fig.text(0.89,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    fig.text(0.89,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
    #We also add a table
    celldata  = [[a1,e1,'{:.1f}'.format(m1/metoms)],['$a_2\in[0.5,10]$',\
                  '$e_2\in'+'[{0:.2f},{1:.2f}]'.format(evals.min(),evals.max())+'$',\
                  '{:.1f}'.format(m2/metoms)]]
    tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
    tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
    
    table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
              loc='top',cellLoc='center')
    
    table.set_fontsize(10)
    
    table.scale(1, 1.2)
    
def max_usefulness_vs_qp(a1=1):

    """Finds the maximum usefulness for each qp value given the (a2,e2)-space
    values we consider. Also plots the a2 values for the cases of 
    qp = 1e-3,1e-2,1e-1,1"""
    
    qp = np.logspace(-3,0,50)
    
    fmax = np.zeros(len(qp))
    fmin = np.zeros(len(qp))
        
    fig, ax = plt.subplots(figsize=(10,6))
    
    for j in range(len(qp)):
        if not j%10:    
            print('Currently on iteration: '+str(j))
        fvals,_ = get_usefulness(a1,m2=qp[j]*m1)
        
        fsum    = np.sum(fvals[0],axis=1)/N_a
        if np.all(fsum==0):
            continue
        fsum[fsum==0] = np.nan
        fmax[j] = np.nanmax(fsum)
        fmin[j] = np.nanmin(fsum)
        
    ax.plot(qp,fmax,'k-',label=r'$\max \langle f_{\mathrm{consumed},a_2} \rangle$')
    ax.plot(qp,fmin,'k--',label=r'$\min \langle f_{\mathrm{consumed},a_2} \rangle$')
    
    #We plot the most useful a-values for each case
    for qpi in [1e-3,1e-2,1e-1,1]:
        
        fvals,_ = get_usefulness(a1,m2=qpi*m1)
        
        fsum  = np.sum(fvals[0],axis=1)/N_a
        if np.all(fsum==0):
            continue
        fsum[fsum==0] = np.nan
        maxid = np.where(fsum==np.nanmax(fsum))[0]
        minid = np.where(fsum==np.nanmin(fsum))[0]
    
        fmaxi = fsum[maxid[0]]
        fmini = fsum[minid[0]]
    
        a2max = avals[maxid[0]]
#        emax = evals[maxid[1]]
        
        a2min = avals[minid[0]]
#        emin = evals[minid[1]]
    
        ax.scatter(qpi,fmaxi,c='k')
        ax.text(qpi,fmaxi,'$a_2 = '+'{:.2f}'.format(a2max)+'\ \mathrm{AU}$',\
                va='bottom',ha='left',fontsize=12)
        ax.scatter(qpi,fmini,c='k',marker='s')
        ax.text(qpi,fmini,'$a_2 = '+'{:.2f}'.format(a2min)+'\ \mathrm{AU}$',\
                va='bottom',ha='left',fontsize=12)
    
    ax.set_xlabel(r'$q_p$')
    ax.set_ylabel(r'$\langle f_{\mathrm{consumed},a_2} \rangle$')
    ax.set_xlim(9e-4,1.1)
    ax.set_ylim(9e-5,1.1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1])
    
#    yticks = ax.yaxis.get_major_ticks()
#    yticks[-1].label1.set_visible(False)
        
    ax.legend(prop={'size':12},loc='upper right')
    
    #We add the current date
    date = datetime.datetime.now()
            
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
            
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    fig.text(0.908,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
    #We also add a table
    celldata  = [[a1,e1,'{:.1f}'.format(m1/metoms)],['$a_2\in[0.5,10]$',\
                  '$e_2\in'+'[{0:.2f},{1:.2f}]'.format(evals.min(),evals.max())+'$',\
                  '{:.1f}'.format(m2/metoms)]]
    tabcol    = [r'$a\ \mathrm{[AU]}$','$e$','$m_p\ [M_\oplus]$']
    tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
    
    table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
              loc='top',cellLoc='center')
    
    table.set_fontsize(10)
    
    table.scale(1, 1.2)

############################## Calling functions #################################

#plt.close('all')

#We first set up our grid of a and e values
avals = np.logspace(-1,1,100)
evals = np.arange(0.5,0.99+0.01,0.01)

N_a = len(avals)
N_e = len(evals)

#We also choose the number of scatterings for each orbital configuration
N_scatter = 1000

#We also select our initial parameters for orbit one

e1 = 0.0
m1 = 300*metoms
#m2 = 1*metoms #Make this an input with the qp factor

theta = 0
#We set up containers for our data
    
usefulness_vs_qp()
#usefulness_vs_qstar(m2=300*metoms)
#plot_usefulness(a1=0.3,m2=metoms,Mstar=1)
#plot_outcome_fractions(a1=0.3,Mstar=10)
#plot_outcome_fractions_qstar(a1=8)
#max_usefulness_vs_qp(a1=0.3)