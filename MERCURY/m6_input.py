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
        
def big_input(names,bigdata,asteroidal=False,epoch=0):
    
    """Function that generates the big.in input file for MERCURY6 given an Nx10
    array of data in the following format:
        
        Columns:
            
            0: mass of the object given in solar masses
            1: radius of the object in Hill radii
            2: density of the object
            3: semi-major axis in AU
            4: eccentricity
            5: inclination in degrees
            6: argument of pericentre in degrees
            7: longitude of the ascending node
            8: mean anomaly in degrees
    
    We can also pass the argument asteroidal as True if we want that coordinate
    system. Also the epoch can be specified, it should be given in years."""
    
    N = len(bigdata)       
    
    if asteroidal:
        style = 'Asteroidal'
    else:
        style = 'Cartesian'
    
    initlist = [')O+_06 Big-body initial data  (WARNING: Do not delete this line!!)\n',\
        ") Lines beginning with `)' are ignored.\n",\
        ')---------------------------------------------------------------------\n',\
        ' style (Cartesian, Asteroidal, Cometary) = {}\n'.format(style),\
        ' epoch (in days) = {}\n'.format(epoch*365.25),\
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
            
def rand_big_input(names,bigdata):
    
    """Function that generates the big.in input file for MERCURY6 given that
    we wish to make a run for an unspecified system. bigdata should be an array
    containing Nx7 array that contains data in the following form:
        
        Columns:
            
            1: mass of the object
            2: radius of the object
            3: density of the object
            4: semi-major axis in AU
            5: eccentricity
            6: inclination
            7: argument of perihelion
            8: longitude of the ascending node
            
    The code generates random properties of the objects from a uniform distribution.
    It yields a new mean anomaly for each body in the system."""
    
    N = len(bigdata)       
    
    initlist = [')O+_06 Big-body initial data  (WARNING: Do not delete this line!!)\n',\
        ") Lines beginning with `)' are ignored.\n",\
        ')---------------------------------------------------------------------\n',\
        ' style (Cartesian, Asteroidal, Cometary) = Asteroidal\n',\
        ' epoch (in days) = 0\n',\
        ')---------------------------------------------------------------------\n']
    
#    ecc = np.random.uniform(0,0.01,size=N)
    
#    i = np.random.uniform(0,5,size=N)
    
#    n = np.random.uniform(0,360,size=N)
    M = np.random.uniform(0,360,size=N)
#    p = np.random.uniform(0,360,size=N)
    
#    bigdata = np.insert(bigdata,4,ecc,axis=1)
#    bigdata = np.insert(bigdata,5,i,axis=1)
#    bigdata = np.insert(bigdata,6,p,axis=1)
#    bigdata = np.insert(bigdata,7,n,axis=1)
    bigdata = np.insert(bigdata,8,M,axis=1)
    
    with open('big.in','w+') as bigfile:
        
        for i in initlist:
            bigfile.write(i)
        
        for j in range(N):
            
            bigfile.write(' {0:11}m={1:.17E} r={2:.0f}.d0 d={3:.2f}\n'.format(names[j],*bigdata[j,0:3]))
            bigfile.write(' {0: .17E} {1: .17E} {2: .17E}\n'.format(*bigdata[j,3:6]))
            bigfile.write(' {0: .17E} {1: .17E} {2: .17E}\n'.format(*bigdata[j,6:]))
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

def setup_end_time(T,T_start=0):
    
    """Small function that sets up the duration of our integration.
        
        T: the total time of the integration given in yr
        T_start: we can also specify the start time. If no start time
            is given, it is set to zero by default. Should aso be given in yr"""
        
    #The string we want to change
    start_str = ' start time (days) = '
    end_str   = ' stop time (days) = '
    
    #Makes temporary file
    fh, abs_path = mkstemp()
    
    with fdopen(fh,'w') as new_file:
        with open('param.in') as old_file:
            for line in old_file:
                if start_str in line:
                    old_sstr = line
    
#                    old_stime = float(old_sstr.strip(start_str))
                    new_stime = T_start
                
                    new_sstr = start_str+str(new_stime)+'\n'
                    new_file.write(line.replace(old_sstr, new_sstr))
                    
                elif end_str in line:
                    old_estr = line
    
                    etime = T*365.25
                    
                    new_estr = end_str+str(etime)+'\n'
                    new_file.write(line.replace(old_estr, new_estr))
                else:
                    new_file.write(line)
    #Remove original file and move new file
    remove('param.in')
    move(abs_path, 'param.in')
    
def setup_rerun_time(T):
    
    """Small function that updates the stop time in param.dmp to allow for an
    extended integration in case we have no collisions. Updates the old time
    value by adding the value T."""
    
    #The string we want to change
    start_str = ' start time (days) = '
    end_str   = ' stop time (days) = '
    
    #Makes temporary file
    fh, abs_path = mkstemp()
    
    with fdopen(fh,'w') as new_file:
        with open('param.in') as old_file:
            lines = old_file.readlines()
            
            for line in lines:
                if start_str in line:
                    old_sstr_idx = lines.index(line)
    
                    
                elif end_str in line:
                    old_estr_idx = lines.index(line)
            
            old_sstr = lines[old_sstr_idx]
            old_estr = lines[old_estr_idx]
            
            old_stime = float(old_sstr.strip(start_str))
            old_etime = float(old_estr.strip(end_str))
            
            new_stime = old_etime
            new_etime = old_etime+T*365.25
            
            new_sstr = start_str+str(new_stime)+'\n'
            new_estr = end_str+str(new_etime)+'\n'
            
            lines[old_sstr_idx] = new_sstr
            lines[old_estr_idx] = new_estr
            
        new_file.writelines(lines)
            
    #Remove original file and move new file
    remove('param.in')
    move(abs_path, 'param.in')
            
def extend_stop_time(T):
    
    """Small function that updates the stop time in param.dmp to allow for an
    extended integration in case we have no collisions. Updates the old time
    value by adding the value T."""
    
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