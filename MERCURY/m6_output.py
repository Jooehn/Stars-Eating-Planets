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

plt.rcParams['font.size']= 16
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 14
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
        if ce_data == True:
                
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

class m6_analysis:
    
    def __init__(self,m6data):
        
        self.autoer = 149597900/6371
        
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
        
    def __get_names(self):
        
        with open('big.in','r') as bigfile:
            
            biglines = bigfile.readlines()
            
        nameid = np.arange(6,len(biglines),4)
            
        self.names = [biglines[i].split()[0] for i in nameid]
                
    def __detect_death(self):
        
        deaths = []
        
        with open('lossinfo.txt','r') as losses:
            
            for line in losses:
                try:
                    simstr  = line.split()[0].split('.')
                    sysid   = simstr[0]
                    simid   = simstr[1]
                    loss    = line.split()[1]
                    t_loss  = line.split()[-2]
                    if loss in self.names:
                        
                        lossid = self.names.index(loss)
                        deaths.append(np.array([sysid,simid,lossid,t_loss],dtype=np.float))  
                        
                except IndexError:
                    continue
                
        return np.asarray(deaths)
    
    def __detect_col(self):
        
        cols = []
        
        with open('lossinfo.txt','r') as losses:
            
            for line in losses:
                try:
                    simstr  = line.split()[0].split('.')
                    sysid   = simstr[0]
                    simid   = simstr[1]
                    loss    = line.split()[1]
                    t_loss  = line.split()[-2]
                    if loss in self.names:
                        
                        lossid = self.names.index(loss)
                        cols.append(np.array([sysid,simid,lossid,t_loss],dtype=np.float))  
                        
                except IndexError:
                    continue
                
        return np.asarray(cols)
    
    def __get_fin_phases(self):
        
        self._dlist = self.__detect_death()
        
        if self.M == 1:
        
            for i in range(self.K):

                if len(self._dlist) == 0:
                    
                    for j in range(self.N):
                    
                        self.fin_phases[i][j] = self.simdata[i][j][-1]
                
                elif i in self._dlist[:,0].astype(int):
                    
                    for j in range(self.N):
                        
                        if j in self._dlist[:,2].astype(int):
                            
                            d_id = np.where((i in self._dlist[:,0].astype(int)) &\
                                     (j in self._dlist[:,2].astype(int)))[0][0]
                            
                            finid = find_nearest(self.simdata[i][j][:,0],self._dlist[d_id,3])
                            
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
                
    def __get_rmin(self):
        
        for i in range(self.K):
            
            rmins = np.zeros((self.N,self.M))
            
            for j in range(self.M):
                
                for k in range(self.N):
                    
                    a = self.simdata[i][j][k][:,1]
                    e = self.simdata[i][j][k][:,2]
                    
                    rmins[k][j] = min(a*(1-e))
                    
            self.rminvals[i][:,0] = np.mean(rmins,axis=1)
            self.rminvals[i][:,1] = np.std(rmins,axis=1)
                
    def Lovis_plot(self):
        
        avals = self.fin_phases[:,:,1]
        evals = self.fin_phases[:,:,2]
        mvals = self.fin_phases[:,:,7]
        tvals = self.fin_phases[:,:,0]
        
        #add function to get tend
        
        tend = get_end_time()
        
        tmask = tvals < tend
        
        msize = mvals**(1/3)*500
        
        avals[tmask] = 0
        msize[tmask] = 0
        
        yvals    = np.zeros((self.N,self.K))
        yvals[:] = np.arange(1,self.K+1,1)
        
        fig, ax = plt.subplots(figsize=(self.N,8))
        
        clist = ['m','olive','g','r','orange']
        
        for i in range(self.N):
            
            peri = avals[:,i]*(1-evals[:,i])
            apo  = avals[:,i]*(1+evals[:,i])
            
            errs = np.asarray([abs(peri-avals[:,i]),abs(apo-avals[:,i])])
            
            _,caps,_  = ax.errorbar(avals[:,i],yvals[i],xerr=errs,fmt='None',ecolor='k',\
                        capsize=4,elinewidth=1.09)
            
            for cap in caps:
                
                cap.set_color(clist[i])
#                cap.set_markeredgewidth(10)
            
            ax.scatter(avals[:,i],yvals[i],s=msize[:,i],c=clist[i],label=self.names[i])
        
        ax.set_yticks(np.arange(1,self.K+1))
        ax.set_xscale('log')
        ax.set_title('$\mathrm{3J+2E\ runs\ evolved\ for\ 0.1\ Myr}$')
        ax.set_ylabel(r'$\mathrm{Run\ index}$')
        ax.set_xlabel('$a\ \mathrm{[AU]}$')
        ax.set_xlim(0.2,200)
        ax.set_ylim(0,self.K+1)
        
#        ax.legend()
        
        fig.canvas.draw()
        
        labels = [item.get_text() for item in ax.get_yticklabels()]
        
        labels = labels
        
        labels = list(range(self.K))
            
        ax.set_yticklabels(labels)
        
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
        
#m6d, ced = m6_load_data(ce_data=True)
m6d = m6_load_data(ce_data=False)
#rrd, ids = find_survivors(m6d)
m6a = m6_analysis(m6d)
#m6a.alpha_vs_teject()
m6a.Lovis_plot()
#dlist = m6a.detect_death()
#jup = m6_output('JUPITER.aei')    
#nep = m6_output('NEPTUNE.aei')