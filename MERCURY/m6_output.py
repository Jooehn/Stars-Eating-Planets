#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:52:27 2019

@author: John Wimarsson
"""

import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import pandas as pd
import math
import string
from plotfuncs import *

plt.rcParams['font.size']= 16
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['mathtext.fontset'] = 'cm'

def is_file_empty(filename):
    with open(filename)as output:
        lines = output.readlines()[4:]
        if len(lines) != 0:
            return False
    return True

def m6_read_output(filenames):
    
    output      = []
    ce_output   = []
    
    for i in filenames:
        
        output.append(np.loadtxt(i+'.aei',skiprows=4))#,dtype=None,encoding=None))
        if not is_file_empty(i+'.clo'):
            ce_output.append(np.genfromtxt(i+'.clo',skip_header=4,dtype=None,encoding=None))
        else:
            ce_output.append(['No recorded close encounters for {}'.format(i)])
        
    return output,ce_output

def m6_save_data(m6data,ce_m6data=[],fname=[]):
    
    if len(fname)==0:
        
        fname = 'm6sim_data.pk1'
        
        ce_fname = 'ce_m6sim_data.pk1'
        
    else:
        
        ce_fname = 'ce_'+fname
    
    cPickle.dump(m6data,open(fname,'wb'))
    
    if len(ce_m6data)!=0:
        
        cPickle.dump(ce_m6data,open(ce_fname,'wb'))
        
def m6_update_data(simdata,simidx,m6data_old):
    
    N = len(m6data_old[0][0])
    
    for i in range(len(simidx.T)):
        
        j, k = simidx[:,i]
        
        for n in range(N):
        
            m6data_old[j][k][n] = np.vstack([m6data_old[j][k][n],simdata[i][n]])
        
    return m6data_old
    
def m6_load_data(filename = None,ce_data=False):
    
    if filename is None:
        if ce_data == True:
            
            fnames = ['m6sim_data.pk1','ce_m6sim_data.pk1']
        else:
            fnames = ['m6sim_data.pk1']
    else:            
        if ce_data:
                
            m6filename = filename[0]
            cefilename = filename[1]
                
            fnames = [m6filename,cefilename]
        else:
            fnames = [filename]
        
    try:
        data = [cPickle.load(open(i,'rb')) for i in fnames]
        
    except FileNotFoundError:
        
        print('No data to be found. You have yet to run a simulation or the file\
               name is wrong.')
    
    if ce_data:
        return data
    else:
        return data[0]
    
def get_end_time():
    
    """Function that finds the end time of the integration and returns it
    in yr unit."""
    
    #The string we want to find
    end_str   = ' stop time (days) = '

    with open('param.in') as old_file:
        for line in old_file:
            if end_str in line:
                estr = line

                etime = float(estr.strip(end_str))
            
    return etime/365.25

def get_Mstar():
    
    """Function that finds the mass of the star in the simulation."""
    
    #The string we want to find
    Mstar_str   = 'central mass (solar) = '

    with open('param.in') as old_file:
        for line in old_file:
            if Mstar_str in line:
                
                Mstar = float(line.split()[-1])
            
    return Mstar
    
def find_survivors(m6data):
    
    """Finds the simulations for which there has been no death of a planet and
    returns the final phases of the planets in said system."""
    
    letters = list(string.ascii_lowercase)
    
    if type(m6data[0][0]) is list:
            
         K = len(m6data)
         M = len(m6data[0])
         N = len(m6data[0][0])

    else:
        
        K = len(m6data)
        M = 1
        N = len(m6data[0])

    letters = letters[:M]
    
    sysidx = []
    simidx = []

    with open('siminfo.txt','r') as siminfo:
        
        simlines = siminfo.readlines()[2:-2]
        
        for line in simlines:
            
            info = line.split()
            
            if int(info[1].strip('.')[0])==N:
            
                idxinfo     = info[0].split('.')
                sysid       = idxinfo[0]
                simid       = letters.index(idxinfo[1])
                
                sysidx.append(int(sysid))
                simidx.append(int(simid))
    
    #Unfinished runs are saved in a list
    
    rrundata = []
    
    for i,j in zip(sysidx,simidx):
        
        fphase = []
        
        simdata = m6data[i][j]
        
        for k in simdata:
            
            fphase.append(k[-1])
            
        rrundata.append(fphase)
        
    return rrundata,np.array([sysidx,simidx])
    
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def check_ce(big_names):
    """Checks the output files and determines if there has been a close encounter or not.
    Returns True for a close encounter and otherwise False."""
    
    ce_bool = np.full(len(big_names),False)
    
    for i in range(len(big_names)):
        
        fname = big_names[i]+'.clo'
        
        if is_file_empty(fname):
            continue
            
        ce_data = np.genfromtxt(big_names[i]+'.clo',skip_header=4,dtype=None,encoding=None)
            
        for line in ce_data:
            
            if line[1] in big_names:
                
                ce_bool[i] = True
                break
                
    return np.any(ce_bool)

class m6_analysis:
    
    def __init__(self,m6data):
        
        """We set up the initial variables of our class. K is the number of 
        simulations, M is the number of sub-simulations and N is the number of 
        planets."""
        
        self.autoer = 149597900/6371
        self.Rs     = 1/215
        self.rcrit  = 3e-2
        self.simdata = m6data
        
        if type(m6data[0][0]) is list:
            
            self.K = len(m6data)
            self.M = len(m6data[0])
            self.N = len(m6data[0][0])
        
            self.fin_phases = np.zeros((self.K,self.M,self.N,8))
            self.rminvals = np.zeros((self.K,self.N,2))
        
        else:
            
            self.K = len(m6data)
            self.M = 1
            self.N = len(m6data[0])
        
            self.fin_phases = np.zeros((self.K,self.N,8))
        
        self.__get_names()
        self.__get_fin_phases()
        self.__find_bad_runs()
        self.__get_init_config()
        
    def __get_init_config(self):
        """Finds the initial phases of the planets in the simulation."""
        
        self._init_phases = np.zeros((self.N,8)) 
        
        for i in range(self.N):
            
            self._init_phases[i] = self.simdata[0][i][0]
        
    def __get_names(self):
        
        with open('big.in','r') as bigfile:
            
            biglines = bigfile.readlines()
            
        nameid = np.arange(6,len(biglines),4)
            
        self.names = [biglines[i].split()[0] for i in nameid]
                
    def __detect_death(self):
        
        deaths = []
        cols   = [[],[],[]]
        with open('lossinfo.txt','r') as losses:
            
            for line in losses:
                try:
                    simid   = line.split()[0]
                    loss    = line.split()[1]
                    t_loss  = line.split()[-2]
                    if loss in self.names:
                        
                        if 'was hit by' in line:
                            tar  = loss
                            loss = line.split()[5]
                            
                            lossid = self.names.index(loss)
                            tarid = self.names.index(tar)
                            
                            cols[0].append(int(simid))
                            cols[1].append(int(lossid))
                            cols[2].append(int(tarid))
                            
                        lossid = self.names.index(loss)
                        deaths.append(np.array([simid,lossid,t_loss],dtype=np.float))  
                except IndexError:
                    continue
        
        self._clist = np.asarray(cols)
        
        return np.asarray(deaths)
    
    def __detect_col(self):
        
        cols = []
        
        with open('lossinfo.txt','r') as losses:
            
            for line in losses:
                simid   = line.split()[0]
                loss    = line.split()[1]
                t_loss  = line.split()[-2]
                if loss in self.names:
                    
                    lossid = self.names.index(loss)
                    cols.append(np.array([simid,lossid,t_loss],dtype=np.float))  
                
        return np.asarray(cols)
    
    def __detect_scol(self):
        
#        self._get_rmin()
        
        self.sclist = [[],[]]
        
        scstr = 'collided with the central body at'
        
        with open('lossinfo.txt','r') as lossfile:
            for line in lossfile:
                
                if scstr in line:
                    sline = line.split()
                    self.sclist[0].append(int(sline[0]))
                    self.sclist[1].append(self.names.index(sline[1]))
    
        self.sclist = np.asarray(self.sclist)
    def __get_fin_phases(self):
        
        self._dlist = self.__detect_death()
        
        self.__detect_scol()
        
        if self.M == 1:
        
            for i in range(self.K):

                if len(self._dlist) == 0:
                    
                    for j in range(self.N):
                    
                        self.fin_phases[i][j] = self.simdata[i][j][-1]
                
                elif i in self._dlist[:,0].astype(int):
                    
                    for j in range(self.N):
                        
                        if j in self._dlist[:,1].astype(int):
                            
                            d_id = np.where((i in self._dlist[:,0].astype(int)) &\
                                     (j in self._dlist[:,1].astype(int)))[0][0]
                            
                            finid = find_nearest(self.simdata[i][j][:,0],self._dlist[d_id,2])
                            
                            self.fin_phases[i][j] = self.simdata[i][j][finid]
                
                else:
                    for j in range(self.N):
                    
                        self.fin_phases[i][j] = self.simdata[i][j][-1]
                    
        else:
            
            for i in range(self.K):
                
                if i in self._dlist[:,0].astype(int):
                
                    for j in range(self.M):
                
                        if j in self._dlist[:,1].astype(int):
                        
                            for k in range(self.N):
                            
                                if k in self._dlist[:,2].astype(int):
                                
                                    d_id = np.where((i in self._dlist[:,0].astype(int)) &\
                                         (j in self._dlist[:,1].astype(int)) &\
                                         (k in self._dlist[:,2].astype(int)))[0][0]
                                
                                    finid = find_nearest(self.simdata[i][j][k][:,0],self._dlist[d_id,3])
                                
                                    self.fin_phases[i][j][k] = self.simdata[i][j][k][finid]
    
                                else:
                                    self.fin_phases[i][j][k] = self.simdata[i][j][k][-1]
    
                        else:
                            for k in range(self.N):    
                                self.fin_phases[i][j][k] = self.simdata[i][j][k][-1]
                                
                else:
                    for j in range(self.M):
                        for k in range(self.N):
                            self.fin_phases[i][j][k] = self.simdata[i][j][k][-1]
                
    def _get_rmin(self):
        
        if self.M>1:
        
            for i in range(self.K):
                
                rmins = np.zeros((self.N,self.M))
                
                for j in range(self.M):
                    
                    for k in range(self.N):
                        
                        a = self.simdata[i][j][k][:,1]
                        e = self.simdata[i][j][k][:,2]
                        
                        rmins[k][j] = min(a*(1-e))
                        
                self.rminvals[i][:,0] = np.mean(rmins,axis=1)
                self.rminvals[i][:,1] = np.std(rmins,axis=1)
        else:
            self.rminvals = np.zeros((self.K,self.N))
            
            for i in range(self.K):
                    
                for j in range(self.N):
                        
                    a = self.simdata[i][j][:,1]
                    e = self.simdata[i][j][:,2]
                        
                    self.rminvals[i][j] = min(a*(1-e))
                    
    def __find_bad_runs(self):
        
        """Detects any runs with an energy loss above the tolerance value."""
        
        etol = 1e-2
        
        self.siminfo = np.loadtxt('siminfo.txt',skiprows=1)
        
        self._badruns = np.where(self.siminfo[:,5]>etol)[0]
        
    def Lovis_plot(self):
        
        avals = self.fin_phases[:,:,1]
        evals = self.fin_phases[:,:,2]
        mvals = self.fin_phases[:,:,7]
        tvals = self.fin_phases[:,:,0]
        
        #add function to get tend
        
        tend = get_end_time()
        
#        tmask = tvals < tend
        
        msize = mvals**(1/3)*500
        
        avals[self._dlist[:,0].astype(int),self._dlist[:,1].astype(int)] = 0
#        avals[tmask]       = 0
#        msize[tmask]       = 0
        
        yvals    = np.zeros((self.N,self.K))
        yvals[:] = np.arange(1,self.K+1,1)
        
        fig, ax = plt.subplots(figsize=(10,self.K/4))
        
        clist = ['m','b','g','r','orange']
        
        #First we plot the original configuration for the planets
        
        init_avals = self._init_phases[:,1]
        
        ax.scatter(init_avals,[self.K+2]*self.N,c=clist,s=msize)
        
        #We also add a dashed line below the initial configuration
        
        ax.axhline(self.K+1,linestyle='-',color='k')
        
        #Next we plot the surviving planets in every run
        
        for i in range(self.N):
            
            nzmask = (avals[:,i]>0)
            
            avalsnz = avals[:,i][nzmask]
            evalsnz = evals[:,i][nzmask]
            yvalsnz = yvals[i][nzmask]
            msizenz = msize[:,i][nzmask]
            
            peri = avalsnz*(1-evalsnz)
            apo  = avalsnz*(1+evalsnz)
            
            errs = np.asarray([abs(peri-avalsnz),abs(apo-avalsnz)])
            
            #The caps indicate the periastron and apoastron of the orbits
            _,caps,_  = ax.errorbar(avalsnz,yvalsnz,xerr=errs,fmt='None',ecolor='k',\
                        capsize=4,elinewidth=1.09)
            
            for cap in caps:
                
                cap.set_color(clist[i])
#                cap.set_markeredgewidth(10)
            
            ax.scatter(avalsnz,yvalsnz,s=msizenz,c=clist[i],label=self.names[i],zorder=2)
            
        scxvals = np.logspace(np.log10(1e-3+4e-4),np.log10(self.Rs-5e-4),5)
        msizesc = mvals**(1/3)*500    
        
        #We then plot the planets that have been scattered into the star
        
        for j in range(self.K):
            if (j in self.sclist[0]) & (j not in self._badruns):
                
                sysid = self.sclist[0][self.sclist[0] == j]
                plaid = self.sclist[1][self.sclist[0] == j]
                
                scx = scxvals[plaid]
                
                scc = [clist[i] for i in plaid]
        
                ax.scatter(scx,sysid+1,s=msizesc[j][plaid],c=scc,zorder=2)
                ax.axhspan((j+1-0.2),(j+1+0.2),alpha=0.3,color='g',zorder=1)
            elif j in self._badruns:
                ax.axhspan((j+1-0.2),(j+1+0.2),alpha=0.3,color='k',zorder=1)
                
        #Finally, we plot the planets that have collided with the survivors
        #behind them
        
        for j in range(self.K):
            if j in self._clist[0]:
                
                sysid = self._clist[0][self._clist[0] == j]
                plaid = self._clist[1][self._clist[0] == j]
                tarid = self._clist[2][self._clist[0] == j]
                
                tarx  = avals[sysid,tarid]
                
                fcc = [clist[i] for i in plaid]
                ecc = [clist[i] for i in tarid]
                
            
                ax.scatter(tarx,sysid+1,s=msizesc[j][tarid],c=fcc,edgecolor=ecc,\
                           linewidth=4,zorder=2)

        ax.set_yticks(np.arange(1,self.K+1))
        ax.set_xscale('log')
        ax.set_title('$\mathrm{3J+2E\ runs\ evolved\ for\ 10\ Myr}$')
        ax.set_ylabel(r'$\mathrm{Run\ index}$')
        ax.set_xlabel('$a\ \mathrm{[AU]}$')
        ax.set_xlim(1e-3,1e4)
        ax.set_ylim(0,self.K+3)
        ax.tick_params(axis='y',which='minor',left=False,right=False)
        ax.axvline(5e-3,linestyle='--',color='k',label=r'$R_\odot$')
        ax.axvline(3e-2,linestyle='--',color='tab:gray',label=r'$r_{crit}$')
        
#        ax.legend()
        
        fig.canvas.draw()
        
        labels = [item.get_text() for item in ax.get_yticklabels()]
        
        labels = labels
        
        labels = list(range(self.K))
            
        ax.set_yticklabels(labels)
        
#        add_date(fig)
        
    def alpha_vs_teject(self):
        
        tend = get_end_time()
        
#        alphavals = np.around((mvals / mvals[0])[:,0],1)

        alphas = np.arange(1.5,4.5,0.5)
        
        #Generalise this script, add ability to extract alphas and tend
        
        alphaej = []
        teject  = []
        tejmin  = []
        tejmax  = []
        
        for i in range(len(alphas)):
            
            if i in self._dlist[:,0].astype(int):
                
                alphaej.append(alphas[i])
                
                ejid = np.where(i == self._dlist[:,0].astype(int))
                
                tval = self._dlist[ejid][:,3]
                
                teject.append(tval.mean())
                tejmin.append(tval.min())
                
                if len(tval) == self.M:
                    tejmax.append(tval.max())
                else:
                    tejmax.append(tend)
        
        teject = np.asarray(teject)/tend
        tejmin = np.asarray(tejmin)/tend
        tejmax = np.asarray(tejmax)/tend
        
        fig, ax = plt.subplots(figsize=(8,6))

        ax.errorbar(alphaej,teject,yerr=np.asarray([teject-tejmin,tejmax-teject]),fmt='ob',ecolor='k',\
                        capsize=2,elinewidth=1.09,uplims=True)        
#        ax.scatter(alphaej,teject,marker='^')

#        ax.set_yscale('log')
        ax.set_xlim(0,5)
        ax.set_ylim(-0.1,2)
        
        ax.set_title('$\mathrm{Duration\ of\ stability\ for\ mass\ boosted\ Solar\ System}$')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\mathrm{Time\ elapsed\ before\ ejection\ [Myr]}$')
        
class m6_ce_analysis:
    
    """Class for analysing close encounters between planets in MERCURY simulations."""
    
    def __init__(self,m6d,ce_data):
        
        self.autoer = 149597900/6371
        self.Rs     = 1/215
        self.rcrit  = 3e-2
        self.simdata = m6d
        self.ce_data = ce_data
        self.Mstar   = get_Mstar()
        
        if type(ce_data[0][0]) is list:
            
            self.K = len(ce_data)
            self.M = len(ce_data[0])
            self.N = len(ce_data[0][0])
        
            self.fin_phases = np.zeros((self.K,self.M,self.N,8))
            self.rminvals = np.zeros((self.K,self.N,2))
        
        else:
            
            self.K = len(ce_data)
            self.M = 1
            self.N = len(ce_data[0])
        
            self.fin_phases = np.zeros((self.K,self.N,8))
        
        self.__get_names()
        
    def __get_names(self):
        
        with open('big.in','r') as bigfile:
            
            biglines = bigfile.readlines()
            
        nameid = np.arange(6,len(biglines),4)
            
        self.names = [biglines[i].split()[0] for i in nameid]
        
    def __get_init_config(self):
        """Finds the initial phases of the planets in the simulation."""
        
        self._init_phases = np.zeros((self.N,8)) 
        
        for i in range(self.N):
            
            self._init_phases[i] = self.simdata[0][i][0]
        
    def __find_bad_runs(self):
        
        """Detects any runs with an energy loss above the tolerance value."""
        
        etol = 1e-2
        
        self.siminfo = np.loadtxt('siminfo.txt',skiprows=1)
        
        self._badruns = np.where(self.siminfo[:,5]>etol)[0]
        
    def __get_first_ce(self):
        
        """Finds the first planet-planet close encounter for each planet"""
        
        first_ce = [[] for i in range(self.N)]
        
        #First we loop through each simulation
        for i in range(self.K):
            
            #Next we loop over each planet
            for j in range(self.N):
                
                #If we have had close encounters for said planet, we proceed
                if len(self.ce_data[i][j])!=0:
                    
                    #We loop through each encounter
                    for k in range(len(self.ce_data[i][j])):
                    
                        ce = self.ce_data[i][j][k]
                        
                        #We save the first recorded planet-planet encounter
                        if ce[1] in self.names:
                            
                            first_ce[j].append(ce)
                            break
        
        self.first_ce = first_ce
        
    def plot_first_ce(self):
        
        """Plots the eccentricity against the semi-major axis of the secondary
        planet in a close encounter. Only accounts for the initial orbit crossing."""
        
        fig, ax = plt.subplots(figsize=(8,6))
        
        #Index zero corresponds to the minor planet
        for i in range(len(self.first_ce[0])):
            
            if len(self.first_ce[0][i])!=0:
                
                a2 = self.first_ce[0][i][6]
                e2 = self.first_ce[0][i][7]
                
                ax.plot(a2,e2,color='b',marker='o')#,ms=10,mew=2)
        
        ax.set_xlabel('$a_2\ \mathrm{[AU]}$')
        ax.set_ylabel('$e_2$')
        ax.set_title('$\mathrm{Secondary\ planet\ orbital\ configuration\ when\ orbits\ cross}$')
        
        ax.set_xlim(0.1,10)
        ax.set_ylim(0,1)
        ax.set_xscale('log')
        
        a1i = self._init_phases[1,1]
        e1i = self._init_phases[1,2]
        m1i = self._init_phases[1,-1]
        a2i = self._init_phases[0,1]
        e2i = self._init_phases[0,2]
        m2i = self._init_phases[0,-1]
        
        q1i = m1i/self.Mstar
        q2i = m2i/self.Mstar
        
        celldata  = [['{:.2f}'.format(a1i),'{:.2f}'.format(e1i),'{:.0e}'.format(q1i)],\
                     ['{:.2f}'.format(a2i),'{:.2f}'.format(e2i),'{:.0e}'.format(q2i)]]
        tabcol    = ['$a\ \mathrm{[AU]}$','$e$','$q$']
        tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
        
        table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
                  loc='top',cellLoc='center')
        
        table.set_fontsize(10)
        
        table.scale(1, 1.2)
        
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)