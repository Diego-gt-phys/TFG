# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:18:11 2025

DANTE: Dark-matter Admixed Neutron-sTar solvEr

This version of DANTE is based on the old Star_Solver_constant_density.py.
It solves the TOV equations for a 1-fluid constant density star. 
Due to this just being a code to compare the accuracy of DANTE to the analytical solution, the only data it saves is the plot.

author: Diego García Tejada
"""

###############################################################################
# Imports and units
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as opt
import multiprocessing as mp
from tqdm import tqdm

# Physical parameters (solar mass = 1.98847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses
rho = 2e-4

###############################################################################
# Define the functions
###############################################################################

###############################################################################
# Define the parameters
###############################################################################
   
###############################################################################
# Calculate the data
###############################################################################

###############################################################################
# Plot the data
###############################################################################