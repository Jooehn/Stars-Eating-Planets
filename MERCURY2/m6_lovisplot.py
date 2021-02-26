#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:45:36 2019

@author: jooehn
"""

from m6_output import *
import os

alpha = 10
title = 'X = {}'.format(alpha)+'\ \mathrm{M}_\oplus'

m6d = m6_load_data(ce_data=False)
    
m6a = m6_analysis(m6d)
m6a.Lovis_plot(title)

os.chdir('../Results')
plt.savefig('2X+3J_alpha_{}.png'.format(alpha),dpi=300)
