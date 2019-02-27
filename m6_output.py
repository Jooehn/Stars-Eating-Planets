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

def m6_save_data(m6data,ce_m6data=[]):
    
    cPickle.dump(m6data,open('m6sim_data.pk1','wb'))
    
    if len(ce_m6data)!=0:
        
        cPickle.dump(ce_m6data,open('ce_m6sim_data.pk1','wb'))
    
def m6_load_data(ce_data=False):
    
    if ce_data == True:
        
        filename = 'ce_m6sim_data.pk1'
    else:
        filename = 'm6sim_data.pk1'
    
    try:
        m6data = cPickle.load(open(filename,'rb'))
        
    except FileNotFoundError:
        
        print('No data to be found. You have yet to run a simulation.')
        
    return m6data      

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

class m6_analysis:
    
    def __init__(self,m6data):
        
        self.simdata = m6data
        
        self.M = len(m6data)
        self.N = len(m6data[0])
        
        self.fin_phases = np.zeros((self.M,self.N,8))
        
        self.__get_fin_phases()
                
    def __detect_death(self):
        
        with open('big.in','r') as bigfile:
            
            biglines = bigfile.readlines()
            
        nameid = np.arange(6,len(biglines),4)
            
        self.names = [biglines[i].split()[0] for i in nameid]
        
        deaths = []
        
        with open('lossinfo.txt','r') as losses:
            
            for line in losses:
                try:
                    sim     = line.split()[0]
                    loss    = line.split()[1]
                    t_loss  = line.split()[-2]
                    if loss in self.names:
                        
                        lossid = self.names.index(loss)
                        deaths.append(np.array([sim,lossid,t_loss],dtype=np.float))  
                        
                except IndexError:
                    continue
                
        return np.asarray(deaths)
    
    def __get_fin_phases(self):
        
        self._dlist = self.__detect_death()
        
        for i in range(self.M):
            
            if i in self._dlist[:,1].astype(int):
                
                for j in range(self.N):
                    
                    if j in self._dlist[:,0].astype(int):
                        
                        d_id = np.where((i in self._dlist[:,1].astype(int)) &\
                                 (j in self._dlist[:,0].astype(int)))[0][0]
                        
                        finid = find_nearest(self.simdata[i][j][:,0],self._dlist[d_id,2])
                        
                        self.fin_phases[i][j] = self.simdata[i][j][finid]
            
            for j in range(self.N):
                
                self.fin_phases[i][j] = self.simdata[i][j][-1]
                
    def Lovis_plot(self):
        
        avals = self.fin_phases[:,:,1]
        mvals = self.fin_phases[:,:,7]
        tvals = self.fin_phases[:,:,0]
        
        tmask = tvals < 1e5
        
        msize = mvals**(1/3)*300
        
        avals[tmask] = 0
        msize[tmask] = 0
        
        yvals    = np.zeros((self.M,self.N))
        yvals[:] = np.arange(1,self.M+1,1)
        
        fig, ax = plt.subplots(figsize=(self.N,8))
        
        alphavals = np.around((mvals / mvals[0])[:,0],1)
        
        clist = ['m','olive','g','r','orange','purple','cyan','blue','grey']
        
        for i in range(self.M):
            
            ax.scatter(avals[:,i],yvals[i],s=msize[:,i],c=clist[i],label=self.names[i])
        
        ax.set_yticks(np.arange(1,self.M+1))
        ax.set_xscale('log')
        ax.set_title('$\mathrm{Mass\ boosted\ Solar\ System\ evolved\ for\ 0.1\ Myr}$')
        ax.set_ylabel(r'$\alpha$')
        ax.set_xlabel('$a\ \mathrm{[AU]}$')
        ax.set_xlim(0.2,200)
        ax.set_ylim(0,self.M+1)
        
#        ax.legend()
        
        fig.canvas.draw()
        
        labels = [item.get_text() for item in ax.get_yticklabels()]
        
        labels = labels
        
        for j in range(len(alphavals)):
            
            labels[j] = '{:.1f}'.format(alphavals[j])
            
        ax.set_yticklabels(labels)
        
m6d = m6_load_data()
m6a = m6_analysis(m6d)
m6a.Lovis_plot()
#dlist = m6a.detect_death()
#jup = m6_output('JUPITER.aei')    
#nep = m6_output('NEPTUNE.aei')