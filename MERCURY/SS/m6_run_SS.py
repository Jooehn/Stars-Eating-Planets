#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:19:24 2019

@author: John Wimarsson
"""

from subprocess import call
from m6_input import *
from m6_output import *
import pandas as pd
import os
import string

##################### Data entry #########################

#Enter data and generate the input files. If smalldata is none, just call
#the function without passing arguments to it

big_names      = ['JUPITER','SATURN','URANUS','NEPTUNE']
small_names    = []

all_names      = big_names+small_names

bigdata = np.loadtxt('SS_data_oe_nt.txt',delimiter=',')

rand_big_input(big_names,bigdata[:,:-1])
#small_input(smalldata)
small_input()

#If we want randomly generated planets, use the following functions instead

#rand_big_input(bigdata)
#rand_small_input(smalldata)

###################### Simulations ##########################

#Here we perform a given number of simulations and save the output data
#for each simulation in arrays and collect said arrays in a list. We also
#create simulation summary file with information such as the fate of our
#integrated bodies and the energy and angular momentum loss for each simulation.

N_big   = len(big_names)
N_small = len(small_names)
N_all   = len(all_names)

#We can also boost the mass of the big objects in the system by a factor alpha

#alpha = np.linspace(1,5,N_sim)
alpha = np.arange(1.5,4.5,0.5)

N_sys = len(alpha)
N_sim = 10

#We define variables for strings that we want to find in the info.out file

frac_ec_int = 'Fractional energy change due to integrator:'
frac_am_ch  = 'Fractional angular momentum change:'
ejected     = 'ejected at'
star_coll   = 'collided with central body at'
coll        = 'was hit by'

event_list  = [ejected,star_coll,coll]

siminfo = np.zeros((N_sim*N_sys,6))

simdata     = []
ce_simdata  = []

#We also set up the duration of each simulation

T = 1e6
setup_stop_time(T)

#We also remove a file that we want to recreate

call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','lossinfo.txt','-delete'])
call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','siminfo.txt','-delete'])

#To label each sub simulation we use the following system: for the first simulation
#given the first alpha value, we name it '0a'. To do so we need letters:

letters = list(string.ascii_lowercase)[:N_sim]

#We loop through each simulation and gather the information we need from it

for k in range(N_sys):
    
    simdata_sys     = []
    ce_simdata_sys  = []
    
    #For one value of alpha, we perform N_sim runs with different phases
    
    for i in range(N_sim):
        
        #The folllwing function randomizes the phase of the big bodies
        rand_big_input(big_names,bigdata)
        
        mass_boost(alpha[k])
        
        #We then want to clear any output or dump files present in the working
        #directory, which we can do via our terminal
        bad_ext = ['*.dmp','*.tmp','*.aei','*.clo','*.out']        
        for j in bad_ext:
            call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name',j,'-delete'])
        
        losslist                = []
        energy_change           = np.array([])
        angmom_change           = np.array([])
        star_collisions         = 0
        ejections               = 0
        collisions              = 0
        
        #We first perform the integration by calling MERCURY
        
        call(['./mercury6'])
        
        #We then want to go through the info.out file, if one of our keywords
        #appear, we note down the details of the input
        
        with open('info.out') as info:
            
            for line in info:
                
                if frac_ec_int in line:
                    ec = float(line.strip(frac_ec_int))
                    energy_change = np.append(energy_change,ec)
                    
                elif frac_am_ch in line:
                    amc = float(line.strip(frac_am_ch))
                    angmom_change = np.append(angmom_change,amc)
                    
                #We then investigate if any special event has occured, if so
                #we write this event to a file that can be used as a reference
                #for post-analysis
                
                elif any([k in line for k in event_list]):
                    
                    if ejected in line:
                        ejections += 1
                    
                    elif star_coll in line:
                        star_collisions += 1
                     
                    elif coll in line:
                        collisions += 1
                
                    losslist.append(line)
                    
        #We can now save the info we have extracted
                    
        survived = N_all-ejections-star_collisions-collisions
        
        with open('lossinfo.txt','a') as lossinfo:
            
            for loss in losslist:
                lossinfo.write('{}.{}\t'.format(k,i)+loss+'\n')
            
        #We now have to save our data into files, that later can be analysed
            
        siminfo[k*N_sim:][i] = np.array([survived,star_collisions,ejections,collisions,max(energy_change),
                                max(angmom_change)])
            
        call(['./element6'])
        call(['./close6'])
            
        orb_data, ce_data = m6_read_output(all_names)            
            
        simdata_sys.append(orb_data)
        ce_simdata_sys.append(ce_data)
        
    #Now the run for our system is complete and we have got the information
    #we wanted. We save the simulation data in each 
    
    simdata.append(simdata_sys)
    ce_simdata.append(ce_simdata_sys)

#With the help of pandas, we can create a nicely formatted file with all the info
#we need to find possible outlier simulations

idx_list = list(range(N_sys))*len(letters)
idx_list.sort()

sim_idx = ['{0}.{1}'.format(k,j) for k,j in zip(idx_list,letters*N_sys)] 

siminfo_pd = pd.DataFrame(siminfo,columns = ['Surv',\
             'SCols','Eject','Cols','dE','dL'])

siminfo_pd.insert(0,'Index',sim_idx)

#We label our simulations using our custom indices

siminfo_pd.set_index('Index',inplace=True)

with open('siminfo.txt','a') as simfile:

    simfile.write(siminfo_pd.__repr__())
    
#We would also like to save all our data, which is efficiently done using a 
#pickle dump
    
if any([len(i)!=0 for i in ce_simdata]):
    
    m6_save_data(simdata,ce_simdata)

else:
    m6_save_data(simdata)
#The data can then easily be extracted using m6_load_data()

print('My work here is done')