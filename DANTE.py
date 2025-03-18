# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 20:47:04 2025

DANTE: Dark-matter Admixed Neutron-sTar solvEr

DANTE is a numerical solver for the Tolman-Oppenheimer-Volkoff (TOV) equations 
applied to a two-fluid star composed of baryonic matter (neutron star) and dark matter, 
assuming only gravitational interaction between the fluids.

Main Features:
- Solves the TOV equation for a two-fluid system.
- Computes Mass-Radius relations for Dark Matter Admixed Neutron Stars (DANS).
- Analyzes the impact of dark matter on neutron star structure.

author: Diego García Tejada
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
Gamma = 5/3
K = 8.016548581726 # Polytropic constant for m = 1GeV, it has units of (km)^2/(solar mass)^(2/3)

###############################################################################
# Define the functions
###############################################################################

def eos_A (p_A): # BM
    """
    Guiven the arrays 'p_data' and 'rho_data' which contain the information for the equation of state, this funtion interpolates the value of rho for a guiven  p. 

    Parameters
    ----------
    p_A : float
        Preassure of fluid A at which we want to evaluate the eos.

    Returns
    -------
    rho : float
        Density associated to the preassure guiven.

    """
    if p_A <= 0:
        return 0
    
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p_A)
    
    return rho

def eos_B (p_B): # DM.
    """
    Polytropic EOS for the star B. 
    
    Parameters:
        p_B (float): Preassure of the fluid B.
        
    Returns:
        float: Density of the fluid B (rho) at preassure p_B.
    """

    if p_B <= 0:
        return 0  # Avoid invalid values
        
    rho = (p_B / K) ** (1/Gamma)
    return rho










