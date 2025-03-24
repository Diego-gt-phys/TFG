# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:54:39 2025

Calculate the data needed for a dm mass analysis.

The analysis consist of a DANS with a fixed amount of baryonic and dark matter.
We vary the mass of the dark matter particles and we see how that changes the structure of the DANS.

@author: Diego García Tejada
"""

###############################################################################
# Imports and units
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d # Needed for interpolation of EoS
import scipy.optimize as opt # Needed to find the values od lambda

# Physical parameters (solar mass = 1.98847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses
PCS = {"soft": (2.785e-6, 5.975e-4), "middle": (2.747e-6, 5.713e-4), "stiff": (2.144e-6, 2.802e-4)} # Central pressure intervals for the MR curves 








