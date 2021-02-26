#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:02:45 2019

@author: John Wimarsson

Script that calls functions and plots the orbital configuration for the secondary
planet in the first close encounter between two planets.
"""

from m6_output import *

m6d, m6ced = m6_load_data(ce_data = True)

m6ce = m6_ce_analysis(m6d,m6ced)

m6ce.plot_first_ce()