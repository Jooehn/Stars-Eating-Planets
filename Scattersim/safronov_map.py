#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:38:11 2019

@author: jooehn
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotfuncs import *
from labellines import labelLines


Msun = 1.9891e33 # 1 solar mass [g]
Mearth = 5.972e27 # 1 earth mass [g]
AU = 1.4959787e13 # 1 astronimical unit [cm]
msuntome = Msun/Mearth
M_s = 1
rjtoau = 1/2150

def safronov_number(mp,ms,ap):
    
    #We use mass radius relation from Tremaine & Dong (2012)
    mp  = mp/msuntome
    mpj = mp*1000 #In Jupiter masses
    rp = 10**(0.087+0.141*np.log10(mpj)-0.171*np.log10(mpj)**2)*rjtoau
    
    saf = np.sqrt(mp*ap/(ms*rp))
    
    return saf

fig,ax = plt.subplots(figsize=(10,6))

masses  = np.linspace(0.1,1000,100)
avals   = np.linspace(0.1,100,100)

xx, yy = np.meshgrid(avals,masses)

safvals = safronov_number(yy,M_s,xx)

levels = np.logspace(np.log10(safvals.min()),np.log10(safvals.max()),100)

contax = ax.contourf(xx,yy,safvals,levels=levels,cmap='plasma',norm=LogNorm())
cont1 = ax.contour(xx,yy,safvals,levels = [1],colors=['w'],linestyles='--',norm=LogNorm())

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)

cbar = plt.colorbar(contax,cax=cax)
cbar.set_label(r'$\Theta$')
cbar.set_ticks([1e-2,1e-1,1e0,1e1])

ax.set_xlabel('$a\ [\mathrm{AU}]$')
ax.set_ylabel(r'$m_p\ [M_\oplus]$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1,100)
ax.set_ylim(0.1,1000)

fmt = {}
c_label = [r'$\Theta = 1$']
for l, s in zip(cont1.levels, c_label):
    fmt[l] = s

ax.clabel(cont1, cont1.levels, inline=True, fmt=fmt, colors='w', fontsize=12,\
          manual=True)

add_date(fig)