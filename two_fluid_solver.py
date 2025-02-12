# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:34:28 2025

@author: Usuario
"""

###############################################################################
# Imports and units
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

# Physical parameters (solar mass = 198847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses

###############################################################################
# Define the functions
###############################################################################

def eos_A (p_A): # The A star has constant density.
    """
    Constant density EOS for the star A.
    
    Parameters:
        p_A (float): Pressure of the fluid A.
    
    Returns:
        float: Density of the fluid A (rho) at preassure p_A.
    """
    
    rho = 4.775e-4
    
    return rho

def eos_B (p_B): # The B star follows a polytropic EOS p=10*rho^(5/3).
    """
    Polytropic EOS for the star B.
    
    Parameters:
        p_B (float): Preassure of the fluid B.
        
    Returns:
        float: Density of the fluid B (rho) at preassure p_B.
    """
    
    K = 10
    gamma = 5/3

    if p_B <= 0:
        return 0  # Avoid invalid values
        
    rho = (p_B / K) ** (1 / gamma)

    return rho

def system_of_ODE (r, y):
    """
    Function that calculates the derivatives of m and the pressures. This function is used for the runge-Kutta method. 

    Parameters
    ----------
    r : float
        radius inside of the star.
    y : tuple
        (m, p_A, p_B), where m is the mass, p_A[p_B] is the preassure of the fluid [B], evaluated at point r.

    Returns
    -------
    dm_dr : float
        rate of change of the mass.
    dpA_dr : float
        rate of change of the pressure of fluid A.
    dpB_dr : float
        rate of change of the preassure of fluid B.

    """
    
    m, p_A, p_B = y
    rho_A = eos_A(p_A)
    rho_B = eos_B(p_B)
    
    dm_dr = 4 * np.pi * (rho_A + rho_B) * r**2
    
    dphi_dr = (G * m + 4 * np.pi * G * r**3 * (p_A + p_B)) / (r * (r - 2 * G * m))
    
    dpA_dr = -(rho_A + p_A) * dphi_dr
    
    dpB_dr = -(rho_B + p_B) * dphi_dr
    
    return (dm_dr, dpA_dr, dpB_dr)
