# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:18:11 2025

DANTE: Dark-matter Admixed Neutron-sTar solvEr

This version of DANTE is based on the old Star_Solver.py.
It solves the TOV equations for a 1-fluid constant density star or polytropic star. 
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

###############################################################################
# Define the functions
###############################################################################
def eos (p, eos_type, rho_0=2e-4, k=10, gamma=2):
    """
    Given a pressure p, gives the value of density rho in acordance to the EoS.
    The function accepts 2 types of EoS: const = Constant density, poly = Polytropic EoS.

    Parameters
    ----------
    p : float
        Pressure.
    eos_type : int
        Type of EoS to use. 0 for an EoS of constant density. 1 for a Polytropic EoS
    rho_0 : float, optional
        Value of denisty in the case of eos_type = 'const'. The default is 2e-4.
    k : float, optional
        If eos_type='poly', value of the polytropic constant. The default is 10.
    gamma : TYPE, optional
        If eos_type='poly', value of the polytropic index. The default is 2.
    Returns
    -------
    TYPE
        Density(p).

    """
    if p <= 0:
        return 0
    
    if eos_type == 0:
        rho = rho_0
    elif eos_type == 1:
        rho = rho = (p / k) ** (1/gamma)
        
    return rho

def system_of_ODE (r, y, eos_type):
    """
    Function that calculates the derivatives of m and the pressure. This function is used for the runge-Kutta method.

    Parameters
    ----------
    r : float
        radius inside of the star.
    y : tuple
        (m, p), where m is the mass, p is the preassure of the fluid, evaluated at point r..
    eos_type : int
        Type of EoS to use. 0 for an EoS of constant density. 1 for a Polytropic EoS.  
    Returns
    -------
    dm_dr : float
        rate of change of the mass.
    dp_dr : float
        rate of change of the pressure of fluid.

    """
    m, p = y
    rho = eos(p, eos_type)
    
    dm_dr = 4 * np.pi * (rho) * r**2
    dphi_dr = (G * m + 4 * np.pi * G * r**3 * (p)) / (r * (r - 2 * G * m))
    dp_dr = -(rho + p) * dphi_dr
    
    return (dm_dr, dp_dr)
    
def RK4O_with_stop (y0, r_range, h, eos_type):
    """
    Function that integrates the y vector using a Runge-Kutta 4th orther method.
    Due to the physics of our problem. The function is built with a condition that doesn't allow negative pressures. 

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_c)
    r_range : tuple
        Range of integratio: (r_0, r_max)
    h : float
        Step size of integration.
    eos_type : int
        Type of EoS to use. 0 for an EoS of constant density. 1 for a Polytropic EoS.
    Returns
    -------
    r_values : array
        Array containing the different values of r.
        
    y_values : array
        Array containig the solutions for the vector y: (m_values, p_values).
    """
    
    r_start, r_end = r_range
    r_values = [r_start]
    y_values = [y0]
    
    r = r_start
    y = np.array(y0)

    while r <= r_end:
        k1 = h * np.array(system_of_ODE(r, y, eos_type))
        k2 = h * np.array(system_of_ODE(r + h / 2, y + k1 / 2, eos_type))
        k3 = h * np.array(system_of_ODE(r + h / 2, y + k2 / 2, eos_type))
        k4 = h * np.array(system_of_ODE(r + h, y + k3, eos_type))

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Stopping conditions: p<p_c*1e-10
        if y_next[1] <= y0[1]*1e-10: # If both pressures drop to 0 then the star stops there.
            break
        
        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next
        
    return (np.array(r_values), np.array(y_values))

def TOV_solver (y0, r_range, h, eos_type):
    """
    Using a 4th Runge Kutta method, it solves the TOV for a perfect fluid star.
    It guives the mass and preassure values in all of the stars radius.

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_c).
    r_range : tuple
        Range of integratio: (r_0, r_max).
    h : float
        Step size of integration.
    eos_type : int
        Type of EoS to use. 0 for an EoS of constant density. 1 for a Polytropic EoS

    Returns
    -------
    r_values : array
        Array containing the different values of r.
    m_values : array
        Array containing the different values of m(r).
    p_values : array
        Array containing the different values of p(r).
    """
    
    r_values, y_values = RK4O_with_stop(y0, r_range, h, eos_type)
    
    m_values = y_values[:, 0]
    p_values = y_values[:, 1]
    
    return (r_values, m_values, p_values)

def find_pc (M_target, eos_type):
    def f(pc):
        r, m, p = TOV_solver((0, pc), (1e-6, 100), 1e-3, eos_type)
    
    
###############################################################################
# Define the parameters
###############################################################################
   
###############################################################################
# Calculate the data
###############################################################################

###############################################################################
# Plot the data
###############################################################################