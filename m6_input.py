#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:51:34 2019

@author: John Wimarsson
"""

import numpy as np
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
##### Format of input #####
# Bigdata must be an array with Nx10 elements containing for each big body the name,
# mass, radius and density, as well its six orbital elements.
# Smalldata must be an array with Nx8 elements, as the small particles are considered
# point objects with not mass, radius or density
        
def big_input(names,bigdata):
    
    """Function that generates the big.in input file for MERCURY6 given an Nx10
    array of data in the following format:
        
        Columns:
            
            0: Name of the object in upper case letters
            1: mass of the object given in solar masses
            2: radius of the object in Hill radii
            3: density of the object
            4: semi-major axis in AU
            5: eccentricity
            6: inclination in degrees
            7: argument of pericentre in degrees
            8: longitude of the ascending node
            9: mean anomaly in degrees"""
    
    N = len(bigdata)       
    
    initlist = [')O+_06 Big-body initial data  (WARNING: Do not delete this line!!)\n',\
        ") Lines beginning with `)' are ignored.\n",\
        ')---------------------------------------------------------------------\n',\
        ' style (Cartesian, Asteroidal, Cometary) = Cartesian\n',\
        ' epoch (in days) = 0\n',\
        ')---------------------------------------------------------------------\n']
    
    with open('big.in','w+') as bigfile:
        
        for i in initlist:
            bigfile.write(i)
        
        for j in range(N):
            
            bigfile.write(' {0:11}m={1:.17E} r={2:.0f}.d0 d={3:.2f}\n'.format(names[j],*bigdata[j,0:3]))
            bigfile.write(' {0: .17E} {1: .17E} {2: .17E}\n'.format(*bigdata[j,3:6]))
            bigfile.write(' {0: .17E} {1: .17E} {2: .17E}\n'.format(*bigdata[j,6:]))
            bigfile.write('  0. 0. 0.\n')
            
def small_input(names=[],smalldata=[],epochs=[]):
    
    """Function that generates the small.in input file for MERCURY6 given an Nx10
    array of data in the following format:
        
        Columns:
            
            0: Name of the object in upper case letters
            1: the object's epoch, set to zero if not relevant
            2: semi-major axis in AU
            3: eccentricity
            4: inclination in degrees
            5: argument of pericentre in degrees
            6: longitude of the ascending node
            7: mean anomaly in degrees
            
    If no data is given, the function will simply write only the necessary lines"""
    
    N = len(smalldata)
    
    if len(epochs) == 0:
        
        epochs = np.zeros(N)
    
    initlist = [')O+_06 Small-body initial data  (WARNING: Do not delete this line!!)\n',\
        ')---------------------------------------------------------------------\n',\
        ' style (Cartesian, Asteroidal, Cometary) = Asteroidal\n',\
        ')---------------------------------------------------------------------\n']
    with open('small.in','w+') as smallfile:
        
        for i in initlist:
            smallfile.write(i)
            
        if N == 0:
            return
        
        for j in range(N):
            
            smallfile.write(' {0:9}epoch={1}\n'.format(*smalldata[j,0:2]))
            smallfile.write('  {0: .17E} {1: .17E} {2: .17E}\n'.format(*smalldata[j,2:5]))
            smallfile.write('  {0: .17E} {1: .17E} {2: .17E}\n'.format(*smalldata[j,5:]))
            smallfile.write('   0. 0. 0.\n')
            
def rand_big_input(bigdata):
    
    """Function that generates the big.in input file for MERCURY6 given that
    we wish to make a run for an unspecified system. bigdata should be an array
    containing Nx7 array that contains data in the following form:
        
        Columns:
            
            0: name of the objects in upper case letters
            1: mass of the object
            2: radius of the object
            3: density of the object
            4: semi-major axis in AU
            5: argument of pericentre in degrees
            
    The code generates random properties of the objects from a uniform distribution.
    It yields eccentricities between 0 and 0.01, inclinations between 0 and 5 degrees,
    longitude of the ascending node between 0 and 360 degrees and mean anomalies
    between 0 and 360 degrees."""
    
    N = len(bigdata)       
    
    initlist = [')O+_06 Big-body initial data  (WARNING: Do not delete this line!!)\n',\
        ')---------------------------------------------------------------------\n',\
        ' style (Cartesian, Asteroidal, Cometary) = Cartesian\n',\
        ' epoch (in days) = 0\n',\
        ')---------------------------------------------------------------------\n']
    
    ecc = np.random.uniform(0,0.01,size=N)
    
    i = np.random.uniform(0,5,size=N)
    
    n = np.random.uniform(0,360,size=N)
    
    M = np.random.uniform(0,360,size=N)
    
    bigdata = np.insert(bigdata,5,ecc,axis=1)
    bigdata = np.insert(bigdata,6,i,axis=1)
    bigdata = np.insert(bigdata,8,n,axis=1)
    bigdata = np.insert(bigdata,9,M,axis=1)
    
    with open('big.in','w+') as bigfile:
        
        for i in initlist:
            bigfile.write(i)
        
        for j in range(N):
            
            bigfile.write(' {0:11}m={1:.17E} r={2:.0f}.d0 d={3:.2f}\n'.format(*bigdata[j,0:4]))
            bigfile.write('  {0: .17E} {1: .17E} {2: .17E}\n'.format(*bigdata[j,4:7]))
            bigfile.write('  {0: .17E} {1: .17E} {2: .17E}\n'.format(*bigdata[j,7:]))
            bigfile.write('   0. 0. 0.\n')
            
def rand_small_input(smalldata,epochs=[]):
    
    """Function that generates the big.in input file for MERCURY6 given that
    we wish to make a run for an unspecified system. smalldata should be an array
    containing Nx2 elements with data in the following form:
        
        Columns:
            
            0: name of the objects in upper case letters
            1: argument of pericentre in degrees
            
    The code generates random properties of the objects from a uniform distribution.
    It yields eccentricities between 0 and 0.01, inclinations between 0 and 5 degrees,
    longitude of the ascending node between 0 and 360 degrees and mean anomalies
    between 0 and 360 degrees. Epochs for the small bodies can be specified
    """
    
    N = len(smalldata)
    
    if len(epochs) == 0:
        
        epochs = np.zeros(N)
    
    a = np.random.uniform(0.65,2,size=N)
    
    ecc = np.random.uniform(0,0.01,size=N)
    
    i = np.random.uniform(0,5,size=N)
    
    n = np.random.uniform(0,360,size=N)
    
    M = np.random.uniform(0,360,size=N)
    
    smalldata = np.insert(smalldata,1,epochs,axis=1)
    smalldata = np.insert(smalldata,2,a,axis=1)
    smalldata = np.insert(smalldata,3,ecc,axis=1)
    smalldata = np.insert(smalldata,4,i,axis=1)
    smalldata = np.insert(smalldata,6,n,axis=1)
    smalldata = np.insert(smalldata,7,M,axis=1)
    
    initlist = [')O+_06 Small-body initial data  (WARNING: Do not delete this line!!)\n',\
        ')---------------------------------------------------------------------\n',\
        ' style (Cartesian, Asteroidal, Cometary) = Asteroidal\n',\
        ' epoch (in days) = 0\n',\
        ')---------------------------------------------------------------------\n']
    with open('small.in','w+') as smallfile:
        
        for i in initlist:
            smallfile.write(i)
        
        if N == 0:
            return
        
        for j in range(N):
            
            smallfile.write(' {0} epoch={1}\n'.format(*smalldata[j,0:2]))
            smallfile.write('  {0: .17E} {1: .17E} {2: .17E}\n'.format(*smalldata[j,2:5]))
            smallfile.write('  {0: .17E} {1: .17E} {2: .17E}\n'.format(*smalldata[j,5:]))
            smallfile.write('   0. 0. 0.\n')
            
def mass_boost(alpha):
    
    """Boosts the mass of the big objects in the system by a factor alpha, which
    is provided as input."""
    
    #Makes temporary file
    fh, abs_path = mkstemp()
    
    with fdopen(fh,'w') as new_file:
        with open('big.in') as old_file:
            for line in old_file:
                if 'm=' in line:
                    
                    #We obtain the old arguments
                    largs = line.split()
    
                    #We extract the mass argument from the file and scale it
                    #by a factor alpha
                    mass_str = largs[1]
    
                    old_mass = float(mass_str.split('m=')[1])
                    new_mass = alpha*old_mass
                    
                    #We then save this as our new mass argument
                    largs[1] = new_mass
                    
                    #Finally we write this new line of object properties into
                    #the big.in file.
                    
                    new_line = (' {0:11}m={1:.17E} {2} {3}\n'.format(*largs))
                    new_file.write(line.replace(line, new_line))
                else:
                    new_file.write(line)                    
    #Remove original file and move new file
    remove('big.in')
    move(abs_path, 'big.in')

def setup_stop_time(T):
    
    """Small function that sets up the duration of our integration.
        
        T: the total time of the integration given in years"""
    
    #The string we want to change
    start_str = 'start time (days) ='
    stop_str = 'stop time (days) ='
    
    #Makes temporary file
    fh, abs_path = mkstemp()
    
    with fdopen(fh,'w') as new_file:
        with open('param.in') as old_file:
            for line in old_file:
                if start_str in line:                    
                    old_time = float(line.strip(start_str))
                    new_file.write(line)
                elif stop_str in line:
                    
                    new_time = T*365.25+old_time
                    
                    new_str =' {0} {1}\n'.format(stop_str,new_time)
                    new_file.write(line.replace(line, new_str))
                else:
                    new_file.write(line)
    #Remove original file and move new file
    remove('param.in')
    move(abs_path, 'param.in')
            
def extend_stop_time(T):
    
    """Small function that updates the stop time in param.dmp to allow for an
    extended integration in case we have no collisions. Updates the old time
    value by adding 30 Myr."""
    
    #The string we want to change
    stime_str = '  stop time (days) =    '
    
    #Makes temporary file
    fh, abs_path = mkstemp()
    
    with fdopen(fh,'w') as new_file:
        with open('param.dmp') as old_file:
            for line in old_file:
                if stime_str in line:
                    old_str = line
    
                    old_time = float(old_str.strip(stime_str))
                    new_time = old_time+T*365.25
                    
                    rep_str = stime_str+str(old_time)
                    new_str = stime_str+str(new_time)
                    new_file.write(line.replace(rep_str, new_str))
                else:
                    new_file.write(line)
    #Remove original file and move new file
    remove('param.dmp')
    move(abs_path, 'param.dmp')
    
mass_boost(1)
            
#bigarr = np.random.uniform(0,9,size=(5,10))
#smallarr = np.random.uniform(0,9,size=(3,8))
#
#rbigarr = np.random.uniform(0,9,size=(5,6))
#rsmallarr = np.random.uniform(0,50,size=(5,2))