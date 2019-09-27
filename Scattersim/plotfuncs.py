#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:32:29 2019

@author: jooehn
"""

import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.size']= 16
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['mathtext.fontset'] = 'cm'

def add_arrow(line, position=None, direction='right', size=14, color=None):
    #Thanks SO
    """Add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken."""
    
    if color is None:
        color = line.get_color()
    
    xvals = line.get_xdata()
    yvals = line.get_ydata()

    x0,x1 = xvals[0],xvals[-1]
    y0,y1 = yvals[0],yvals[-1]
    
    xdata = np.linspace(x0,x1,100)
    ydata = np.linspace(y0,y1,100)

    if position is None:
        position = xdata.mean()
    # find closest index
    if position == x1:
        start_ind = -2
    else:
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
    
def add_date(fig,xcoord=0.88):
    """Adds a box with the current date in the upper right corner of
    the figure"""
    date = datetime.datetime.now()
    
    datestr = '${0}$-${1}$-${2}$'.format(date.day,date.month,date.year)
    
    fig.text(xcoord,0.945,datestr,bbox=dict(facecolor='None'),fontsize=14)
    
def add_AUax(ax,scale,log=True):
    """Adds a new y-axis on the right hand side of the axes which is given
    in AU."""
    axn = ax.twinx()
    
    xdata = ax.lines[0].get_xdata()
    ydata = ax.lines[0].get_ydata()
    ylab  = ax.get_ylabel()
    
    ylablist = ylab.split()
    unit = ylablist[-1]
    
    ylab = ylablist.remove(unit)
    
    ylabn = ''.join(ylablist)+' [\mathrm{AU}]$'
    
    axn.plot(xdata,ydata*scale,color='none')
    axn.set_ylabel(ylabn)
    axn.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks())))
    
    if log:
        axn.set_yscale('log')