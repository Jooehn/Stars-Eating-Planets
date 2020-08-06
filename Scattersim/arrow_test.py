#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:26:07 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt

def add_arrow(line, position=None, direction='right', size=15, color=None):
        """
        add an arrow to a line.
    
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.
        """
        
#        try:
        if color is None:
            color = line.get_color()
    
        xdata = line.get_offsets()[:,0]
        ydata = line.get_offsets()[:,1]
            
        if position is None:
            position = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1
    
        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size
        )
        
fig, ax = plt.subplots()        
        
#t = np.linspace(-2,2,100)
t = np.array([0.5,0.6])
y = np.sin(t)

line, = ax.plot(t,y)

add_arrow(line)

plt.show()