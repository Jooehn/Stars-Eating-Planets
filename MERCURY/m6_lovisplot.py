#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:45:36 2019

@author: jooehn
"""

from m6_output import *

try:
    m6d
except NameError:
    m6d = m6_load_data(ce_data=False)
    
#m6d, ced = m6_load_data(filename=fnames,ce_data=True)
#rrd, ids = find_survivors(m6d)
m6a = m6_analysis(m6d)
#m6a.alpha_vs_teject()
m6a.Lovis_plot()
#dlist = m6a.detect_death()
#jup = m6_output('JUPITER.aei')    
#nep = m6_output('NEPTUNE.aei')