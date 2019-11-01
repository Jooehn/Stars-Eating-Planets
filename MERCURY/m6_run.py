#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:19:24 2019

@author: John Wimarsson
"""

from subprocess import call
from subprocess import Popen
from m6_input import *
from m6_output import *
import yagmail
import pandas as pd
import os
import string

##################### Simulation setup #########################

#Enter data and generate the input files. If smalldata is none, just call
#the function without passing arguments to it

big_names      = ['X1','X2','J1','J2','J3']
small_names    = []

all_names      = big_names+small_names

bigdata = np.loadtxt('2X+3J.data',delimiter=',')

rand_big_input(big_names,bigdata)
#small_input(smalldata)
small_input()

#If we want randomly generated phases, use the following functions instead

#rand_big_input(bigdata)
#rand_small_input(smalldata)

#There is also an argument that if True only keeps the run going until we have
#had at least one host star-planet collision.

until_scol = False
until_ce   = False

#If we want to continue an already terminated run we can use the continue bool
#and set it to True

shall_continue = True

#We also choose the number of simulations
N_sim0 = 0
N_sim  = 50

#We also set up the duration of each simulation. T is the total allowed time for
#the integration while dt is the time allocated for every sub integration. After
#dt we check if any planet has hit the host star. If so we terminate the integration
#otherwhise, we extend the time by dt.

T   = 1e7
dt  = 1e5
setup_end_time(dt)

###################### Simulations ##########################

#Here we perform a given number of simulations and save the output data
#for each simulation in arrays and collect said arrays in a list. We also
#create simulation summary file with information such as the fate of our
#integrated bodies and the energy and angular momentum loss for each simulation.

N_big   = len(big_names)
N_small = len(small_names)
N_all   = len(all_names)

#We define variables for strings that we want to find in the info.out file

frac_ec_int = 'Fractional energy change due to integrator:'
frac_am_ch  = 'Fractional angular momentum change:'
ejected     = 'ejected at'
star_coll   = 'collided with the central body at'
coll        = 'was hit by'

event_list  = [ejected,star_coll,coll]

#Here we either load data or create new containers

if shall_continue:
    
    simdata, ce_simdata = m6_load_data(ce_data = True)
    siminfo = np.loadtxt('siminfo.txt',skiprows=1,usecols=[1,2,3,4,5,6])
    #We check if there is non-registered simulation information
    for i in range(len(siminfo)):
        if not np.any(siminfo[i]):
            lastid = i
            break
    if len(simdata) != lastid:
        
        orb_data, ce_data = m6_read_output(all_names)            
    
        simdata.append(orb_data)
        ce_simdata.append(ce_data)
    
        call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','*.pk1','-delete'])
    
        if any([len(i)!=0 for i in ce_simdata]):
        
            m6_save_data(simdata,ce_simdata)
    
        else:
            m6_save_data(simdata)
                         
    N_sim0 = lastid
            
else:
    
    siminfo = np.zeros((N_sim,6))
    simdata     = []
    ce_simdata  = []
    call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','lossinfo.txt','-delete'])

#We loop through each simulation and gather the information we need from it

for k in range(N_sim0,N_sim):
        
    #The following function randomizes the phase of the big bodies
    rand_big_input(big_names,bigdata)
    
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
    
    #We now carry out our sub-integrations for which we check if we have had
    #any star-planet collisions
    t = 0
    
    while t < T:
        
        #We first perform the integration by calling MERCURY
        
        call(['./mercury6'])    
        
        if until_scol:
        
            #If we only want to run until we have had at least one S-P collision
            #we check the output info to catch such an event and breaks the loop
            #if we find one.
            with open('info.out') as info:
                
                infolines = info.readlines()
                
            for line in infolines:
                
                if star_coll in line:
                    break
            #If we don't find it, we extend the stop time by dt.
            else:
                extend_stop_time(dt)
                t += dt
                continue
            break
        
        elif until_ce:
            
            #If we only want to run until we have a close encounter we enter
            #this statement
            call(['./close6'])
            if check_ce(big_names):
                call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','*.clo','-delete'])
                break
            #If we don't find it, we extend the stop time by dt.
            else:
                call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','*.clo','-delete'])
                extend_stop_time(dt)
                t += dt
        else:
            extend_stop_time(dt)
            t+=dt
    
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
            lossinfo.write('{}\t'.format(k)+loss+'\n')
        
    #We now have to save our data into files, that later can be analysed
        
    siminfo[k] = np.array([survived,star_collisions,ejections,collisions,max(abs(energy_change)),
                            max((angmom_change))])
        
    #Now the run for our system is complete and we have got the information
    #we wanted. We save the simulation data in each 
    
    call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','siminfo.txt','-delete'])

    #With the help of pandas, we can create a nicely formatted file with all the info
    #we need to find possible outlier simulations
    
    siminfo_pd = pd.DataFrame(siminfo,columns = ['Surv',\
                 'SCols','Eject','Cols','dE','dL'])
    
    with open('siminfo.txt','a') as simfile:
    
        simfile.write(siminfo_pd.to_string())
        
    #We obtain the output data for the run
    
    call(['./element6'])
    call(['./close6'])
    
    orb_data, ce_data = m6_read_output(all_names)            
    
    simdata.append(orb_data)
    ce_simdata.append(ce_data)
    
    call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name','*.pk1','-delete'])
    
    #We would also like to save all our data, which is efficiently done using a 
    #pickle dump
    
    if any([len(i)!=0 for i in ce_simdata]):
        
        m6_save_data(simdata,ce_simdata)
    
    else:
        m6_save_data(simdata)

#The data can then easily be extracted using m6_load_data()

#We then want to clear any output or dump files present in the working
#directory, which we can do via our terminal
bad_ext = ['*.dmp','*.tmp','*.aei','*.clo','*.out']        
for j in bad_ext:
    call(['find',os.getcwd(),'-maxdepth','1','-type','f','-name',j,'-delete'])
    
#We send an email to indicate that the run is finished
    
email = 'nat15jwi@student.lu.se'
yag = yagmail.SMTP(email,'D3f3nd1m3px4')
yag.send(email,'MERCURY Run','The run is now finished')

print('My work here is done')
