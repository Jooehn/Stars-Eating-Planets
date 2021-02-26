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
from matplotlib.patches import Rectangle

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

def get_masses():
    """Finds the masses of the planets in the run"""
    
    with open('big.in','r') as bigfile:
    
        biglines = bigfile.readlines()
            
    nameid = np.arange(6,len(biglines),4)
            
    masses = [biglines[i].split()[1].strip('m=') for i in nameid]
    
    return np.asarray(masses,dtype=float)

def get_rho():
    """Finds the masses of the planets in the run"""
    
    with open('big.in','r') as bigfile:
    
        biglines = bigfile.readlines()
            
    nameid = np.arange(6,len(biglines),4)
            
    rho = [biglines[i].split()[3].strip('d=') for i in nameid]
    
    return np.asarray(rho,dtype=float)
    
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
    
def check_gladman_unstable(a1,a2):
    """Checks the Gladman stability criterion for a system"""    

    Mstar = get_Mstar()
    
    m1, m2 = get_masses()    
    
    Lo_mut = ((m1+m2)/(3*Mstar))**(1/3)*(a1+a2)*0.5
    
    glad_unst = abs(a2-a1)/Rhill_mut <= 2*np.sqrt(3)
    
    return glad_unst

def check_ce(big_names):
    """Checks the output files and determines if there has been a close encounter or not.
    Returns True for a close encounter and otherwise False."""
    
    ce_bool = np.full(len(big_names),False)
    
    Mstar = get_Mstar()
    
    m1i, m2i = get_masses()
    
    for i in range(len(big_names)):
        
        fname = big_names[i]+'.clo'
        
        if is_file_empty(fname):
            continue
            
        ce_data = np.genfromtxt(big_names[i]+'.clo',skip_header=4,dtype=None,encoding=None)
        
        if ce_data.size==1:
            ced_list = ce_data.tolist()
            if ced_list[1] in big_names:        
                
                try:
                    a1i = float(ced_list[3])
                    a2i = float(ced_list[6])
                except ValueError:
                    continue
            
                Rhill_mut = ((m1i+m2i)/(3*Mstar))**(1/3)*(a1i+a2i)*0.5
                
                if ced_list[2]<=Rhill_mut:    
                    ce_bool[i] = True
                
        elif ce_data.size>1:
            for line in ce_data:
                
                if line[1] in big_names:
                    
                    try:
                        a1i = float(line[3])
                        a2i = float(line[6])
                    except ValueError:
                        break
            
                    Rhill_mut = ((m1i+m2i)/(3*Mstar))**(1/3)*(a1i+a2i)*0.5
                
                    if line[2] <= Rhill_mut:
                        
                        ce_bool[i] = True
                        break
                
    return np.any(ce_bool)

class m6_analysis:
    
    def __init__(self,m6data,ce_data=None):
        
        """We set up the initial variables of our class. K is the number of 
        systems, M is the number of realisations for each system and N is the 
        number of planets."""
        
        self.autoer = 149597900/6371
        self.cmtoau = 6.68458712e-14
        self.Rs     = 1/215
        self.Rj     = 5.68278912e-04
        self.rcrit  = 3e-2
        self.simdata = m6data
        self.ce_data = ce_data
        
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
        self.__detect_death()
        self.__get_fin_phases()
        self.__find_bad_runs()
        self.__get_init_config()
        self._get_rmin()
        
        if ce_data is not None:
            
            self.ceK = len(ce_data)
            self.ceM = 1
            self.ceN = len(ce_data[0])
            
            self.coll_bool = np.full(len(ce_data),False)
            for i in range(len(ce_data)):
            
                if type(ce_data[i][0]) == np.ndarray:
                    self.coll_bool[i] = True  
                elif type(ce_data[i][0][0]) == np.void:
                    self.coll_bool[i] = True
                elif type(ce_data[i][0][0]) == np.ndarray:
                    self.coll_bool[i] = True
        
    def __get_init_config(self):
        """Finds the initial phases of the planets in the simulation."""
        
        self._init_phases = np.zeros((self.K,self.N,8)) 
        
        for i in range(self.K):
        
            for j in range(self.N):
                
                self._init_phases[i][j] = self.simdata[i][j][0]
        
    def __get_names(self):
        
        with open('big.in','r') as bigfile:
            
            biglines = bigfile.readlines()
            
        nameid = np.arange(6,len(biglines),4)
            
        self.names = [biglines[i].split()[0] for i in nameid]
                
    def __detect_death(self):
        
        deaths = []
        cols   = [[],[],[]]
        sclist = [[],[]]
        ejlist = [[],[]]
        with open('lossinfo.txt','r') as losses:
            
            for line in losses:
                try:
                    sline = line.split()
                    simid   = sline[0]
                    loss    = sline[1]
                    t_loss  = sline[-2]
                    if loss in self.names:
                        
                    #We go through each loss line and give an integer
                    #identifyer for each death type    
        
                        if 'collided with the central body' in line:
                            
                            outcome = 0
                            
                            sclist[0].append(int(sline[0]))
                            sclist[1].append(self.names.index(sline[1]))
                            
                        elif 'ejected at' in line:
                            
                            outcome = 1
                            
                            ejlist[0].append(int(sline[0]))
                            ejlist[1].append(self.names.index(sline[1]))
                        
                        elif 'was hit by' in line:
                            
                            outcome = 2
                            
                            tar  = loss
                            loss = sline[5]
                            
                            lossid = self.names.index(loss)
                            tarid = self.names.index(tar)
                            
                            cols[0].append(int(simid))
                            cols[1].append(int(lossid))
                            cols[2].append(int(tarid))
                            
                        lossid = self.names.index(loss)
                        deaths.append(np.array([simid,lossid,t_loss,outcome],dtype=np.float))  
                except IndexError:
                    continue
        
        self._clist  = np.asarray(cols)
        self._ejlist = np.asarray(ejlist)
        self.sclist  = np.asarray(sclist)
        
        self._dlist  = np.asarray(deaths)
    
    def __get_fin_phases(self):
        
        if self.M == 1:
        
            for i in range(self.K):
                
                #We check if the given run has had any deaths
                
                Kmask = i == self._dlist[:,0].astype(int)
                
                if np.any(Kmask):
                    
                    #If so we look through each planets fate. If it has died
                    #we save the correct final output
                    for j in range(self.N):
                        
                        if len(Kmask) > 1:
                            
                            Nmask = self._dlist[Kmask][:,1] == j
                        else:
                            Nmask = self._dlist[Kmask][1] == j
                        
                        if np.any(Nmask):
                            
                            d_id = np.where((i == self._dlist[:,0].astype(int)) &\
                                     (j == self._dlist[:,1].astype(int)))[0][0]
                            
                            finid = find_nearest(self.simdata[i][j][:,0],self._dlist[d_id,2])
                            
                            self.fin_phases[i][j] = self.simdata[i][j][finid]
                            
                        else:
                            
                            self.fin_phases[i][j] = self.simdata[i][j][-1]
                
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
        
        """Finds the minimum distance between the host star and the planets
        during the entirety of a simulation."""
        
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
            self._rminvals = np.zeros((self.K,self.N))
            self.rminvals = np.zeros((self.K,self.N))
            self.rmaxvals = np.zeros((self.K,self.N))
            self.aminvals = np.zeros((self.K,self.N))
            self.amaxvals = np.zeros((self.K,self.N))
            
            for i in range(self.K):
                    
                for j in range(self.N):
                    
                    mask = i == self._dlist[:,0]
                    
                    a = self.simdata[i][j][:,1]
                    e = self.simdata[i][j][:,2]
                    
                    if j in self._dlist[:,1][mask]:
                        self.rminvals[i][j] = 0
                        self.rmaxvals[i][j] = 0
                        self.aminvals[i][j] = 0
                        self.amaxvals[i][j] = 0
                        
                    else:
                        self.rminvals[i][j] = min(a[a>0]*(1-e[a>0]))
                        self.rmaxvals[i][j] = max(a[a>0]*(1+e[a>0]))
                        self.aminvals[i][j] = min(a[a>0])
                        self.amaxvals[i][j] = max(a[a>0])
                    
                    self._rminvals[i][j] = min(a[a>0]*(1-e[a>0]))
                    
    def __find_bad_runs(self):
        
        """Detects any runs with an energy loss above the tolerance value."""
        
        etol = 1e-2
        
        self.siminfo = np.loadtxt('siminfo.txt',skiprows=1)
        
        self._badruns = np.where(self.siminfo[:,5]>etol)[0]
        
    def _get_M_consumed(self):
        
        """Returns an array containing the number of eaten planets for each
        run index."""
        
        self.M_consumed = np.zeros(self.K)
        
        mvals = self.fin_phases[:self.K,:,7]
        
        for i in range(self.K):
            
            plaid = self.sclist[1][self.sclist[0] == i]
                
            self.M_consumed[i] += mvals[i][plaid].sum()
            
    def _calc_R_Roche(self):
        
        """Calculates the Roche limit for the lowest density planet in the system"""
        
        k = 1.6
        
        M_s = 1.99*10**33
        
        rho = min(get_rho()) #in g/cm^3
        
        #Assuming that both bodies have a uniform density distribution
        
        d = (9*M_s/(4*np.pi*rho))**(1/3)
        
        return d*self.cmtoau
    
    def _get_N_ce(self):
        
        """Finds the number of close encounters for a planet that has died
        during the evolution of its system. Also creates an array with all
        the final encounters before death."""
        
        self._N_ce = np.zeros((self.K,self.N))
            
        dsysid = self._dlist[:,0]
        
        self._final_ce = []
        
        for i in range(len(dsysid)):
            
            sysid = self._dlist[i,0].astype(int)
            
            plaid = self._dlist[i,1].astype(int)
            
            tdeath = self._dlist[i,2].astype(float)
            
            outcome = self._dlist[i,3].astype(int)
                
            ced = self.ce_data[sysid][plaid]
            
            if type(ced) == list:
                
                self._first_ce.append(np.nan)
                self._final_ce.append(np.nan)
                    
                self._N_ce[sysid][plaid] = 0
                
                continue
            
            elif ced.shape == ():
                
                self._N_ce[sysid][plaid] = 1
                self._first_ce.append(ced.tolist())
                self._final_ce.append(ced.tolist())
            
                continue
            
            tfin = ced[-1][0]
            
            fcei = 0
            Nce  = 1
            
            if tfin > tdeath:

                for j in range(len(ced)):
                    
                    if np.isclose(ced[j][0],tdeath):
                        fcei = j
                        Nce  = j+1
                        break
                    
                if outcome == 2:
                    
                    self._N_ce[sysid][plaid] = Nce
                    self._final_ce.append(ced[fcei])
                    continue

                edeath = ced[fcei][4]
                
                while edeath>=1:
                    
                    fcei -= 1
                    Nce  -= 1
                    
                    edeath = ced[fcei][4]
                    
                self._N_ce[sysid][plaid] = Nce
                self._final_ce.append(ced[fcei])
                
            else:
                self._N_ce[sysid][plaid] = len(ced)
                self._final_ce.append(ced[-1])
                
    def _get_first_ce(self):
        
        """Loops through the close encounter data for all system and planets
        to find the first encounter for each system planet, if any."""
        
        self._first_ce = [] #Contains a list for each system with all N first ce's
        self._t_fce = np.zeros((self.K,self.N))
        
        for i in range(self.K):
            
            fce_sys = []
            
            for j in range(self.N):
        
                ced = self.ce_data[i][j]
                    
                #We check if there has been no encounters
                if type(ced) == list:
                    
                    fce = np.nan
                    self._t_fce[i][j] = fce
                    fce_sys.append(fce)
                    
                    continue
                
                #We check if there has been only one encounter
                elif ced.shape == ():
                    
                    fce = ced.tolist()
                    tfce = fce[0]
                    self._t_fce[i][j] = tfce
                    fce_sys.append(fce)
                    
                    continue
                
                #Otherwise we proceed as normally
                fce = ced[0]
                tfce = fce[0]
                self._t_fce[i][j] = fce[0]
            
                fce_sys.append(fce)
                
            self._first_ce.append(fce_sys)
        
    def Lovis_plot(self,title,N_runs = None):
        
        #Set up limits for axes
        
        aminlim = 1e-3
        amaxlim = 1e4
        
        if N_runs is None:
            N_runs = self.K
        
        avals = self.fin_phases[:N_runs,:,1]
        evals = self.fin_phases[:N_runs,:,2]
        mvals = self.fin_phases[:N_runs,:,7]
        tvals = self.fin_phases[:N_runs,:,0]
        
        tend = get_end_time()
        
        msize = mvals**(1/3)*500
        
        if self._dlist.size > 0:  
            mask = self._dlist[:,0] < N_runs
            avals[self._dlist[mask,0].astype(int),self._dlist[mask,1].astype(int)] = 0
        
        yvals    = np.zeros((self.N,N_runs))
        yvals[:] = np.arange(1,N_runs+1,1)
        
        if N_runs/4 > 8:
            figheight = N_runs/4
        else:
            figheight = N_runs/2
        
        fig, ax = plt.subplots(figsize=(10,figheight))
        
        clist = ['m','b','g','r','orange']
        
        #First we plot the original configuration for the planets
        
        init_avals = self._init_phases[0,:,1]
        
        ax.scatter(init_avals,[N_runs+2]*self.N,c=clist,s=msize)
        
        #We also add a line below the initial configuration
        
        ax.axhline(N_runs+1,linestyle='-',color='k')
        
        #We now sort the values based on how close they got to the host star
        #during the simulation and 
        
        #This is done creating an indice list using np.lexsort
        
        #We sort according to:
        #1. Total mass consumed by host star
        #2. Radial excursions towards the host star
        
        self._get_M_consumed()
        
        rmin = self.rminvals
        rmin[rmin == 0] = np.nan
        rmax = self.rmaxvals
        rmax[rmax == 0] = np.nan
        amin = self.aminvals
        amin[amin == 0] = np.nan
        amax = self.amaxvals
        amax[amax == 0] = np.nan
        
        rmins = np.nanmin(rmin,axis=1)
        rmaxs = np.nanmax(rmax,axis=1) 
        amins = np.nanmin(amin,axis=1)
        amaxs = np.nanmax(amax,axis=1) 
        sortindex = np.lexsort((1-rmins,self.M_consumed))
        
        #Next we plot the surviving planets in every run

        for i in range(self.N):
            
            nzmask = (avals[sortindex,i]>0)
            
            avalsnz = avals[sortindex,i][nzmask]
            evalsnz = evals[sortindex,i][nzmask]
            yvalsnz = yvals[i][nzmask]
            msizenz = msize[sortindex,i][nzmask]
            
            peri = avalsnz*(1-evalsnz)
            apo  = avalsnz*(1+evalsnz)
            
            errs = np.asarray([abs(peri-avalsnz),abs(apo-avalsnz)])
            
            #The caps indicate the periastron and apoastron of the orbits
            _,caps,_  = ax.errorbar(avalsnz,yvalsnz,xerr=errs,fmt='None',ecolor='k',\
                        capsize=4,elinewidth=1.09,zorder=1)
            
            for cap in caps:
                
                cap.set_color(clist[i])
#                cap.set_markeredgewidth(10)
            
            ax.scatter(avalsnz,yvalsnz,s=msizenz,c=clist[i],label=self.names[i],zorder=2)
            
        scxvals = np.logspace(np.log10(1e-3+4e-4),np.log10(self.Rs-5e-4),self.N)
        msizesc = mvals**(1/3)*500
        
        #We then plot the planets that have been scattered into the star
        #The consuming systems are marked with a green bar inside of R_star
        #The shade of green depends on the mass consumed
        
        scalpha = np.zeros(len(self.M_consumed))
        
        Mc_unique = np.unique(self.M_consumed)
        
        alphavals = np.linspace(0.15,0.65,len(Mc_unique))
        
        for i in range(len(Mc_unique)):
            
            alphamask = self.M_consumed == Mc_unique[i]
            
            scalpha[alphamask] = alphavals[i]
            
        for k in range(N_runs):
            
            j = sortindex[k]
            if (j in self.sclist[0]):
                
                sysid = self.sclist[0][self.sclist[0] == j]
                plaid = self.sclist[1][self.sclist[0] == j]
                
                scx = scxvals[plaid]
                
                scc = [clist[i] for i in plaid]
        
                ax.scatter(scx,[k+1]*len(scx),s=msizesc[j][plaid],c=scc,zorder=2)
                if j in self._badruns:
                    ax.axhspan((k+1-0.2),(k+1+0.2),alpha=0.3,color='k',zorder=1)
                else:
                    greenxvals = np.linspace(aminlim,self.Rs,100)
                    ax.fill_between(greenxvals,(k+1-0.2),(k+1+0.2),alpha=scalpha[j],color='g',zorder=0)
            elif j in self._badruns:
                ax.axhspan((k+1-0.2),(k+1+0.2),alpha=0.3,color='k',zorder=1)
            
                
            rxvals = np.linspace(rmins[j],rmaxs[j],100)
            axvals = np.linspace(amins[j],amaxs[j],100)
                
            ax.fill_between(rxvals,(k+1-0.2),(k+1+0.2),alpha=0.3,color='r',zorder=1)
            ax.fill_between(axvals,(k+1-0.2),(k+1+0.2),alpha=0.3,color='b',zorder=1)
                
        #Finally, we plot the planets that have undergone a merger 
        
        for k in range(N_runs):
            
            j = sortindex[k]
            
            if j in self._clist[0]:
                
                sysid = self._clist[0][self._clist[0] == j]
                plaid = self._clist[1][self._clist[0] == j]
                tarid = self._clist[2][self._clist[0] == j]
                
                tarx  = avals[sysid,tarid]
                
                tarx[tarx == 0] = scxvals[tarid[tarx == 0]]
                
                fcc = [clist[i] for i in plaid]
                ecc = [clist[i] for i in tarid]
                
                ax.scatter(tarx,[k+1]*len(tarx),s=msizesc[j][tarid],c=fcc,edgecolor=ecc,\
                           linewidth=0.06*msizesc[j][tarid],zorder=2)

        ax.set_yticks(np.arange(1,N_runs+1))
        ax.set_xscale('log')
        ax.set_title(r'$\mathrm{2X+3J\ evolved\ for\ 10\ Myr,\ '+title+'}$')
        ax.set_ylabel(r'$\mathrm{System\ index}$')
        ax.set_xlabel('$a\ \mathrm{[AU]}$')
        ax.set_xlim(aminlim,amaxlim)
        ax.set_ylim(0,N_runs+3)
        ax.tick_params(axis='y',which='minor',left=False,right=False)
        
        #We add a vertical line indicating the radius of the host star
        ax.axvline(self.Rs,linestyle='--',color='k',label=r'$R_\odot$')
        
        #We add two other lines indicating the Roche limit for each planet
        RR = self._calc_R_Roche()
        
        ax.axvline(RR,linestyle='--',color='tab:gray',zorder=1)
        
        #We add an additional line representing the semi-major axis of a Jupiter
        #that has ejected two other Jupiters
        
        af = 1/(1/5+1/7.74+1/11.98)
        ax.axvline(af,linestyle='dotted',color='k',zorder=3)
        
#        ax.legend()
        
        fig.canvas.draw()
        
        labels = [item.get_text() for item in ax.get_yticklabels()]
        
        labels = labels
        
        labels = list(range(N_runs))
            
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
        
    def plot_ce_vs_rmin(self,title):
        
        """Detects the number of close encounters for the planets in the system"""
        
        if not np.any(self.coll_bool):
            raise ValueError('No close encounters have been recorded')
                    
        fig, ax = plt.subplots()
        
        clist = ['m','b','g','r','orange']
        
        N_ce = np.zeros((self.K,self.N))
        
        rmin = self._rminvals
        
        rmin[rmin==0] = 1e-3
        
        self._get_N_ce()
        
        N_ce = self._N_ce
        
        #We use different markers depending on the planet's fate
        
        for i in range(len(self._dlist[:,0])):
            
            plaid = list(range(self.N))
                
            dplaid =  self._dlist[:,1][self._dlist[:,0]==i]
            
            plaid = [j for j in plaid if j not in dplaid]
            
            #We first plot planets that have been consumed
            if i in self.sclist[0]:
                
                scid = self.sclist[1][self.sclist[0]==i]
                
                scc = [clist[j] for j in scid]
                
                ax.scatter(rmin[i][scid],N_ce[i][scid],c=scc,marker='*')
                
                dplaid = [j for j in dplaid if j not in scid]
                
            #Next, we plot merged planets
            if i in self._clist[0]:
                
                cid = self._clist[1][self._clist[0]==i]
                tid = self._clist[2][self._clist[0]==i]
                
                cc = [clist[j] for j in cid]
                ecc = [clist[j] for j in tid]
                
                ax.scatter(rmin[i][cid],N_ce[i][cid],c=cc,edgecolor=ecc,marker='X',\
                           linewidth=1.25)
                
                dplaid = [j for j in dplaid if j not in cid]
                
            #We also plot ejected planets
            if len(dplaid)!=0:
                
                eid = [int(j) for j in dplaid]
                
                ec = [clist[j] for j in eid]
                
                ax.scatter(rmin[i][eid],N_ce[i][eid],c=ec,marker='^')
                
        ax.set_xscale('log')
        ax.set_yscale('symlog')
        ax.set_title(r'$\mathrm{2X+3J\ evolved\ for\ 10\ Myr,\ '+title+'}$')
        ax.set_ylabel(r'$N_\mathrm{ce}$')
        ax.set_xlabel('$r_\mathrm{min}\ \mathrm{[AU]}$')
        ax.set_xlim(9e-4,2e1)
        ax.set_ylim(-0.25,1e4)
#        ax.tick_params(axis='y',which='minor',left=False,right=False)
        
        #We add a vertical line indicating the radius of the host star
        ax.axvline(5e-3,linestyle='--',color='k',label=r'$R_\odot$')
        
    def plot_e1_vs_e2(self,title):
        
        fig, ax = plt.subplots()
        
        clist = ['m','b','g','r','orange']
        
        self._get_N_ce()
        
        dsysid = self._dlist[:,0].astype(int) 
            
        for i in  range(len(dsysid)):
            
            sysid = self._dlist[i,0].astype(int)
            
            plaid = self._dlist[i,1].astype(int)
            
            tdeath = self._dlist[i,2].astype(float)
            
            outcome = self._dlist[i,3].astype(int)
            
            col = clist[plaid]
            
            #We check if the planet was consumed by the host star
            if outcome == 0:
            
                try:
                    e1 = self._final_ce[i][4]
                    e2 = self._final_ce[i][7]
#                    cetime  = self._final_ce[i][0]
#                    tarname = self._final_ce[i][1]
#                    tarid   = self.names.index(tarname)
#                    tid     = find_nearest(self.simdata[sysid][tarid][:,0],cetime)
#                    pid     = find_nearest(self.simdata[sysid][plaid][:,0],cetime)
#                
#                    if not np.isclose(self.simdata[sysid][plaid][pid,0],cetime):
#                        if self.simdata[sysid][plaid][pid,0] > cetime:
#                            pid -= 1
#                    if not np.isclose(self.simdata[sysid][tarid][tid,0],cetime):
#                        if self.simdata[sysid][plaid][pid,0] > cetime:
#                            tid -= 1
#                    e1 = self.simdata[sysid][plaid][pid,2]
#                    e2 = self.simdata[sysid][tarid][tid,2]
                except TypeError:
                    pass
                
                ax.scatter(e1,e2,c=col,marker='*')
                
            #Or if it was ejected
            elif outcome == 1:
                
                try:
                    e1 = self._final_ce[i][4]
                    e2 = self._final_ce[i][7]
#                    cetime  = self._final_ce[i][0]
#                    tarname = self._final_ce[i][1]
#                    tarid   = self.names.index(tarname)
#                    tid     = find_nearest(self.simdata[sysid][tarid][:,0],cetime)
#                    pid     = find_nearest(self.simdata[sysid][plaid][:,0],cetime)
#                
#                    if not np.isclose(self.simdata[sysid][plaid][pid,0],cetime):
#                        if self.simdata[sysid][plaid][pid,0] > cetime:
#                            pid -= 1
#                    if not np.isclose(self.simdata[sysid][tarid][tid,0],cetime):
#                        if self.simdata[sysid][plaid][pid,0] > cetime:
#                            tid -= 1
#            
#                    e1 = self.simdata[sysid][plaid][pid,2]
#                    e2 = self.simdata[sysid][tarid][tid,2]
                except TypeError:
                    pass
                
                ax.scatter(e1,e2,c=col,marker='^')
                
            #Or if the planet died from merger
            elif outcome == 2:
                
                #As the phases of the planets that have merged in the close encounter 
                #data is not entirely trustworthy, we use a different approach here
                
                lpid = np.where((sysid==self._clist[0]) & (plaid==self._clist[1]))
            
                #We find the eccentricity of the target planet before collision
                tarid = int(self._clist[2][lpid])
                tid = find_nearest(self.simdata[sysid][tarid][:,0],tdeath)
                pid = find_nearest(self.simdata[sysid][plaid][:,0],tdeath)
                
                if not np.isclose(self.simdata[sysid][plaid][pid,0],tdeath):
                    if self.simdata[sysid][plaid][pid,0] > tdeath:
                        pid -= 1
                if not np.isclose(self.simdata[sysid][tarid][tid,0],tdeath):
                    if self.simdata[sysid][plaid][pid,0] > tdeath:
                        tid -= 1
            
#                e1 = self.fin_phases[sysid,plaid,2]
                e1 = self.simdata[sysid][plaid][pid,2]
                e2 = self.simdata[sysid][tarid][tid,2]
                
                ecc = clist[tarid]
                
                ax.scatter(e1,e2,c=col,edgecolor=ecc,linewidth=1.25,marker='X')
        
#        ax.set_xscale('symlog',linthreshx=5e-2)
#        ax.set_yscale('symlog',linthreshy=5e-2)
#        ax.set_xticks([i*1e-1 for i in range(11)])
#        ax.set_yticks([i*1e-1 for i in range(11)])
        ax.set_title(r'$\mathrm{2X+3J\ evolved\ for\ 10\ Myr,\ '+title+'}$')
        ax.set_xlabel('$e_\mathrm{CME}$')
        ax.set_ylabel('$e_\mathrm{survivor}$')
        ax.set_xlim(0,1.1)
        ax.set_ylim(0,1.1)
        
        ticks = np.arange(0,1+0.2,0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        sticks = ['$'+str(np.around(i,1))+'$' for i in np.arange(0,1+0.2,0.2)]
        ax.set_xticklabels(sticks)
        ax.set_yticklabels(sticks)
        
    def plot_init_e(self,title,plaid = None):
        
        if plaid is None:
            plaid = np.arange(0,self.N,1)
        
        clist = ['m','b','g','r','orange']
        
        cols  = [clist[i] for i in plaid]
        
        #We sort according to:
        #1. Total mass consumed by host star
        #2. Radial excursions towards the host star
        
        self._get_M_consumed()
        
        rmin = self.rminvals
        rmin[rmin == 0] = np.nan
        
        rmins = np.nanmin(rmin,axis=1)
        sortindex = np.lexsort((1-rmins,self.M_consumed))
        
        evals   = self._init_phases[:,plaid,2][sortindex]
        mvals   = self._init_phases[:,plaid,7][sortindex]
        sysids  = np.arange(0,self.K,1)
        
        msize = mvals**(1/3)*500
        
        if self.K/4 > 8:
            figheight = self.K/4
        else:
            figheight = self.K/2
        
        fig, ax = plt.subplots(figsize=(10,figheight))

        for i in range(len(plaid)):
            ax.scatter(evals[:,i],sysids,c=cols[i],s=msize[:,i],zorder=2)
            
        scalpha = np.zeros(len(self.M_consumed))
        
        Mc_unique = np.unique(self.M_consumed)
        
        alphavals = np.linspace(0.15,0.65,len(Mc_unique))
        
        for i in range(len(Mc_unique)):
            
            alphamask = self.M_consumed == Mc_unique[i]
            
            scalpha[alphamask] = alphavals[i]
            
        xvals = np.linspace(0,1e-2,100)
        
        for k in range(self.K):
            j = sortindex[k]
            if (j in self.sclist[0]):
                ax.fill_between(xvals,(k-0.2),(k+0.2),alpha=scalpha[j],color='g',zorder=0)
        
        ax.minorticks_off()
        ax.set_yticks(sysids)
        ax.set_title(r'$\mathrm{2X+3J\ evolved\ for\ 10\ Myr,\ '+title+'}$')
#        ax.set_xscale('log',subsx=[2, 3, 4, 5, 6, 7, 8, 9])
        ax.tick_params(axis='x', which='minor', bottom=True)
        ax.set_xlim(0,1e-2)
        ax.set_ylim(-1,self.K+1)
        ax.set_ylabel('$\mathrm{System\ index}$')
        ax.set_xlabel('$e_\mathrm{init}$')
        
        fig.canvas.draw()
        
        labels = [item.get_text() for item in ax.get_yticklabels()]
        
        labels = labels
        
        labels = list(range(self.K))
        
        ax.set_yticklabels(labels)
        
    def plot_first_ce(self,title,plaid=None):
        
        self._get_first_ce()
        
        if plaid is None:
            plaid = np.arange(0,self.N,1)
        
        clist = ['m','b','g','r','orange']
        
        cols  = [clist[i] for i in plaid]
        
        #We sort according to:
        #1. Total mass consumed by host star
        #2. Radial excursions towards the host star
        
        self._get_M_consumed()
        
        rmin = self.rminvals
        rmin[rmin == 0] = np.nan
        
        rmins = np.nanmin(rmin,axis=1)
        sortindex = np.lexsort((1-rmins,self.M_consumed))
        
        mvals   = self._init_phases[:,plaid,7][sortindex]
        sysids  = np.arange(0,self.K,1)
        
        msize = mvals**(1/3)*500
        
        #We set up the figure
        
        if self.K/4 > 8:
            figheight = self.K/4
        else:
            figheight = self.K/2
        
        fig, ax = plt.subplots(figsize=(10,figheight))
            
        #Next we determine the first close encounter for each system
        
        tfce = np.nanmin(self._t_fce,axis=1)[sortindex]
        
        ax.scatter(tfce,sysids,s=msize,c=cols)
        
###############################################################################
###############################################################################
############################ Close Encounters##################################
###############################################################################
###############################################################################        
        
        
class m6_ce_analysis:
    
    """Class for analysing close encounters between planets in MERCURY simulations."""
    
    def __init__(self,m6d,ce_data):
        
        self.autoer = 149597900/6371
        self.Rs     = 1/215
        self.rcrit  = 3e-2
        self.simdata = m6d
        self.ce_data = ce_data
        self.Mstar   = get_Mstar()
        
        self.__get_names()
        
        #We check if we have had any collisions at all
    
        coll_bool = np.full(len(ce_data),False)
        for i in range(len(ce_data)):
            
            if type(ce_data[i][0]) == np.ndarray:
                coll_bool[i] = True  
            elif type(ce_data[i][0][0]) == np.void:
                coll_bool[i] = True
            elif type(ce_data[i][0][0]) == np.ndarray:
                coll_bool[i] = True
                
        if not np.any(coll_bool):
            raise ValueError('No close encounters have been recorded')
        
#        if type(ce_data[coll_bool][0][0]) == list:
#            
#            self.K = len(ce_data)
#            self.M = len(ce_data[0])
#            self.N = len(ce_data[0][0])
#        
#            self.fin_phases = np.zeros((self.K,self.M,self.N,8))
#            self.rminvals = np.zeros((self.K,self.N,2))
#        
#        else:
            
        self.K = len(ce_data)
        self.M = 1
        self.N = len(ce_data[0])
    
        self.fin_phases = np.zeros((self.K,self.N,8))
        
        self.__get_first_ce()
        self.__get_init_config()
        self.__calc_Rhill()
        
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
        
    def __calc_Rhill(self):
        
        a1i = self._init_phases[1,1]
        e1i = self._init_phases[1,2]
        m1i = self._init_phases[1,-1]
        a2i = self._init_phases[0,1]
        e2i = self._init_phases[0,2]
        m2i = self._init_phases[0,-1]
        
        self.Rhill_mut = ((m1i+m2i)/(3*self.Mstar))**(1/3)*(a1i+a2i)*0.5
        
    def __get_first_ce(self):
        
        """Finds the first planet-planet close encounter for each planet"""
        
        first_ce = [[] for i in range(self.N)]
        
        m1i, m2i = get_masses()
        
        #First we loop through each simulation
        for i in range(self.K):
            
            #Next we loop over each planet
            for j in range(self.N):
                
                #If we have had close encounters for said planet, we proceed
                if type(self.ce_data[i][j])==list:
                    continue
                
                if self.ce_data[i][j].size>1:
                    #We loop through each encounter
                    for k in range(len(self.ce_data[i][j])):
                    
                        ce = self.ce_data[i][j][k]
                        
                        #We save the first recorded planet-planet encounter
                        if ce[1] in self.names:
                            
                            #We make sure that we disregard any cases where
                            #the system has been ejected from the system which
                            #gives '*****' in the output
                            try:
                                a1i = float(ce[3])
                                a2i = float(ce[6])
                            except ValueError:
                                break
                    
                            Rhill_mut = ((m1i+m2i)/(3*self.Mstar))**(1/3)*(a1i+a2i)*0.5
                            
                            if ce[2]<=Rhill_mut:
                                first_ce[j].append(ce)
                                break
                elif self.ce_data[i][j].size==1:

                    ce = self.ce_data[i][j].tolist()                      
                    #We save the first recorded planet-planet encounter
                    if ce[1] in self.names:
                        
                        try:
                            a1i = float(ce[3])
                            a2i = float(ce[6])
                        except ValueError:
                            continue
                    
                        Rhill_mut = ((m1i+m2i)/(3*self.Mstar))**(1/3)*(a1i+a2i)*0.5
                        
                        if ce[2]<=Rhill_mut:
                            first_ce[j].append(ce)
        
        self.first_ce = first_ce
        
    def plot_first_ce(self):
        
        """Plots the eccentricity against the semi-major axis of the secondary
        planet in a close encounter. Only accounts for the initial orbit crossing."""
        
        fig, ax = plt.subplots(figsize=(10,8))
        
        #Index zero corresponds to the minor planet
        for i in range(len(self.first_ce[0])):
            
            if len(self.first_ce[0][i])!=0:
                
                a2 = self.first_ce[0][i][6]
                e2 = self.first_ce[0][i][7]
                
                if float(e2) < 1:
                
                    ax.plot(a2,e2,color='b',marker='o')#,ms=10,mew=2)
        
        ax.set_xlabel('$a_2\ \mathrm{[AU]}$')
        ax.set_ylabel('$e_2$')
        ax.set_title('$\mathrm{Secondary\ planet\ orbital\ configuration\ at\ first\ orbit\ crossing}$',pad=50.0)
        
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
        tabcol    = ['$a_{init}\ \mathrm{[AU]}$','$e_{init}$','$q$']
        tabrow    = ['$\mathrm{Orbit\ 1}$','$\mathrm{Orbit\ 2}$']
        
        table = ax.table(cellText=celldata,colLabels=tabcol,rowLabels=tabrow,\
                  loc='top',cellLoc='center')
        
        table.set_fontsize(10)
        
        table.scale(1, 1.2)
        
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
