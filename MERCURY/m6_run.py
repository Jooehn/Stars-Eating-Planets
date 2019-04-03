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

##################### Data entry #########################

#Enter data and generate the input files. If smalldata is none, just call
#the function without passing arguments to it

big_names      = ['MERCURY','VENUS','EARTHMOO','MARS','JUPITER','SATURN',\
                  'URANUS','NEPTUNE','PLUTO']
small_names    = []

all_names      = big_names+small_names

#big_input(bigdata)
#small_input(smalldata)
small_input()

#If we want randomly generated planets, use the following functions instead

#rand_big_input(bigdata)
#rand_small_input(smalldata)

#We can also boost the mass of the big objects in the system by a factor alpha

#alpha = 2
#mass_boost(alpha)

###################### Simulations ##########################

#Here we perform a given number of simulations and save the output data
#for each simulation in arrays and collect said arrays in a list. We also
#create simulation summary file with information such as the fate of our
#integrated bodies and the energy and angular momentum loss for each simulation.

N_big   = len(big_names)
N_small = len(small_names)
N_all   = len(all_names)

N_sim = 1

#We define variables for strings that we want to find in the info.out file

frac_ec_int = 'Fractional energy change due to integrator:'
frac_am_ch  = 'Fractional angular momentum change:'
ejected     = 'ejected at'
star_coll   = 'collided with central body at'
rem_a       = 'removed due to small a'
rem_ce      = 'removed due to an encounter with'

event_list  = [ejected,star_coll,rem_ce,rem_a]

siminfo = np.zeros((N_sim,6))

simdata     = []
ce_simdata  = []

#We also set up the duration of each simulation

T = 1000
setup_stop_time(T)

#We also remove some files that we want to recreate

call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','lossinfo.txt','-delete'])

#We loop through each simulation and gather the information we need from it

for i in range(N_sim):
    
    N_runs = 0
    
    while True:
    
        #We then want to clear any output or dump files present in the working
        #directory, which we can do via our terminal
        bad_ext = ['*.dmp','*.tmp','*.aei','*.clo','*.out']        
        for j in bad_ext:
            call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name',j,'-delete'])
        
        energy_change           = np.array([])
        angmom_change           = np.array([])
        star_collisions         = 0
        ejections               = 0
        collisions              = 0
        removed                 = 0
        
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
                
                elif any([i in line for i in event_list]):
                    
                    if ejected in line:
                        ejections += 1
                    
                    elif star_coll in line:
                        star_collisions += 1
                     
                    elif any([rem_a in line,rem_ce in line]):
                        removed += 1
                
                    with open('lossinfo.txt','a') as lossinfo:
                        lossinfo.write('{}\t'.format(i)+line+'\n')
                    
        #We can now save the info we have extracted
                    
        survived = N_all-ejections-star_collisions-removed

        #We only want to consider runs where the energy change due to the integrator
        #is insignificantly small, if the energy change is larger than 1e-4 we
        #therefore reiterate
        
        if abs(max(energy_change))<1e-4:
            
            #If all bodies survive, we extend the integration, else we move on and
            #save the data
    
            if survived < N_all:
                break
            else:
                N_runs +=1
                print('No casualties in simulation {0}. Integrating for another {1} yr'.format(i,T))
                extend_stop_time(T)
                if N_runs == 3:
                    print('All planets survived in simulation {}, even after integrating for an'.format(i)\
                          +' extra {} yr'.format(3*T))
                    break
        
    #We now have to save our data into files, that later can be analysed
        
    siminfo[i] = np.array([survived,star_collisions,ejections,removed,max(energy_change),
                        max(angmom_change)])
    
    call(['./element6'])
    call(['./close6'])
    
    orb_data, ce_data = m6_read_output(all_names)            
    
    simdata.append(orb_data)
    ce_simdata.append(ce_data)
    
    #Now the run is complete and we have got the information we wanted
    
siminfo_pd = pd.DataFrame(siminfo,columns = ['Survived','Star Collisions','Ejections','Removed','dE','dL'])

#With the help of pandas, we can create a nicely formatted file with all the info
#we need to find possible outlier simulations
    
with open('siminfo.txt','w+') as simfile:
    
    simfile.write(siminfo_pd.__repr__())

#We would also like to save all our data, which is efficiently done using a 
#pickle dump
    
if any([len(i)!=0 for i in ce_simdata]):
    
    m6_save_data(simdata,ce_simdata)

else:
    m6_save_data(simdata)
#The data can then easily be extracted using m6_load_data()

print('My work here is done')