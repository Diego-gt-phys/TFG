# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:34:28 2025

Solves the Tolman Oppenheimer Volkoff (TOV) equation for a star formed of two fluids A and B.

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
    
    if p_A <= 0:
        return 0  # Avoid invalid values
    
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

def RK4O_with_stop (y0, r_range, h):
    """
    Function that integrates the y vector using a Runge-Kutta 4th orther method.
    Due to the physics of our problem. The function is built with a condition that doesn't allow negative pressures. If both of them are 0 then the integration stops. 

    Parameters
    ----------
    y0 : tuple
        Stating conditions for our variables: (m_0, p_A_c, P_B_c)
    r_range : tuple
        Range of integratio: (r_0, r_max)
    h : float
        Step size of integration.

    Returns
    -------
    r_values : array
        Array containing the different values of r.
        
    y_values : array
        Array containig the solutions for the vector y.

    """
    
    r_start, r_end = r_range
    r_values = [r_start]
    y_values = [y0]
    
    r = r_start
    y = np.array(y0)

    while r <= r_end:
        k1 = h * np.array(system_of_ODE(r, y))
        k2 = h * np.array(system_of_ODE(r + h / 2, y + k1 / 2))
        k3 = h * np.array(system_of_ODE(r + h / 2, y + k2 / 2))
        k4 = h * np.array(system_of_ODE(r + h, y + k3))

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Stopping conditions: p<p_c*1e-10
        if y_next[1] < y0[1]*1e-10:  # If fluid A's pressure drops to 0, keep it that way
            y_next[1] = 0
            
        if y_next[2] < y0[2]*1e-10:  # If fluid B's pressure drops to 0, keep it that way
            y_next[2] = 0
            
        if y_next[1] < y0[1]*1e-10 and y_next[2] < y0[2]*1e-10: # If both pressures drop to 0 then the star stops there.
            break

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next

    return (np.array(r_values), np.array(y_values))