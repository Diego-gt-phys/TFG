# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:15:49 2025

Solves the TOV equation for Dark matter (fluid B) Admixed Neutron Star (fluis A).

This version instead of working with the alpha parameter for determinig the central pressure of te dark matter it will work with a lambda parameter.

This lambda parameter is calculated by M_B/M.

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
K = 92.29

###############################################################################
# Define the functions
###############################################################################

def eos_A (p_A): # The fluid A is the Neutron matter
    """
    Guiven the arrays p_data and rho_data which contain the information for the equation of state, this funtion interpolates the value of rho for a guiven  p. 

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

def eos_B (p_B): # The fluid B is the dark matter.
    """
    Polytropic EOS for the star B.
    
    Parameters:
        p_B (float): Preassure of the fluid B.
        
    Returns:
        float: Density of the fluid B (rho) at preassure p_B.
    """

    if p_B <= 0:
        return 0  # Avoid invalid values
        
    rho = (p_B / K) ** (3 / 5) # 35
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
    
    m, p_A, p_B, m_A, m_B = y
    rho_A = eos_A(p_A)
    rho_B = eos_B(p_B)
    
    dm_dr = 4 * np.pi * (rho_A + rho_B) * r**2
    
    dmA_dr = 4 * np.pi * (rho_A) * r**2
    
    dmB_dr = 4 * np.pi * (rho_B) * r**2
    
    dphi_dr = (G * m + 4 * np.pi * G * r**3 * (p_A + p_B)) / (r * (r - 2 * G * m))
    
    dpA_dr = -(rho_A + p_A) * dphi_dr
    
    dpB_dr = -(rho_B + p_B) * dphi_dr
    
    return (dm_dr, dpA_dr, dpB_dr, dmA_dr, dmB_dr)

def RK4O_with_stop (y0, r_range, h):
    """
    Function that integrates the y vector using a Runge-Kutta 4th orther method.
    Due to the physics of our problem. The function is built with a condition that doesn't allow negative pressures. If both of them are 0 then the integration stops. 

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_A_c, P_B_c, m_A_0, m_B_0)
    r_range : tuple
        Range of integratio: (r_0, r_max)
    h : float
        Step size of integration.

    Returns
    -------
    r_values : array
        Array containing the different values of r.
        
    y_values : array
        Array containig the solutions for the vector y: (m_values, p_A_values, P_B_values).
        
    R_A : float
        Value of the radius of the star of fluid A
    """
    
    r_start, r_end = r_range
    r_values = [r_start]
    y_values = [y0]
    
    r = r_start
    y = np.array(y0)
    
    R_A = 0

    while r <= r_end:
        k1 = h * np.array(system_of_ODE(r, y))
        k2 = h * np.array(system_of_ODE(r + h / 2, y + k1 / 2))
        k3 = h * np.array(system_of_ODE(r + h / 2, y + k2 / 2))
        k4 = h * np.array(system_of_ODE(r + h, y + k3))

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Stopping conditions: p<p_c*1e-10
        if y_next[1] <= y0[1]*1e-10 and y_next[2] <= y0[2]*1e-10: # If both pressures drop to 0 then the star stops there.
            break
        
        elif y_next[1] <= y0[1]*1e-10 and R_A == 0:  # If fluid A's pressure drops to 0, keep it that way
            y_next[1] = 0
            R_A = r
            print("The star has a Halo of Dark Matter")
            
        elif y_next[1] <= y0[1]*1e-10 and R_A != 0:  # If fluid A's pressure drops to 0, keep it that way
            y_next[1] = 0
            
        elif y_next[2] <= y0[2]*1e-10:  # If fluid B's pressure drops to 0, keep it that way
            y_next[2] = 0

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next
        
    if R_A == 0:
        R_A = r_values[-1]
        print("The star has a core of Dark Matter")
        
    return (np.array(r_values), np.array(y_values), R_A)

def TOV_solver(y0, r_range, h):
    """
    Using a 4th Runge Kutta method, it solves the TOV for a 2 perfect fluid star.
    It guives the mass and preassure values for both fluids in all of the stars radius.

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_A_c, P_B_c, m_A_0, m_B_0)
    r_range : tuple
        Range of integration: (r_0, r_max)
    h : float
        Step size of integration.

    Returns
    -------
    r_values : array
        Array containing the different values of r.
    m_values : array
        Array containing the different values of m(r).
    p_A_values : array
        Array containing the different values of P_A(r).
    p_B_values : array
        Array containing the different values of P_B(r).
    R_A : float
        Value of the radius of the star of fluid A
    """
    
    r_values, y_values, R_A = RK4O_with_stop(y0, r_range, h)
    
    m_values = y_values[:, 0]
    p_A_values = y_values[:, 1]
    p_B_values = y_values[:, 2]
    m_A_values = y_values[:, 3]
    m_B_values = y_values[:, 4]

    return (r_values, m_values, p_A_values, p_B_values, m_A_values, m_B_values, R_A)

def MR_curve(pc_range, alpha, r_range, h, n):
    """
    Creates the mass radius curve of a family of 2-fluid stars by solving the TOV equations.
    The different masses are calculating by changing the central pressure of fluid A and mantaining the alpha value of fluid B. 

    Parameters
    ----------
    pc_range : tuple
        Range of central pressures to integrate: (pc_start, pc_end)
    alpha : float
        Relationship between the central pressure of fluid B and fluid A: pc_B = alpha * pc_A
    r_range : tuple
        Range of radius integration: (r_0, r_max)
    h : float
        integration step.
    n : int
        Number of datapoints of the curve.

    Returns
    -------
    R_values : array
        Array containig the radius of the stars.
    M_values : array
        Array containing the total masses of the stars.
    MA_values : array
        Array containing the mass of fluid A of the stars.
    MB_values : array
        Array containing the mass of fluid B of the stars.
    """
    pc_start, pc_end = pc_range
    pc_list = np.geomspace(pc_start, pc_end, n) 
    R_values = []
    M_values = []
    MA_values = []
    MB_values = []
    cont = 0
    for pc in pc_list:
        r_i, m_i, p_A, p_B, m_a, m_b, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), r_range, h)
        
        R_i = R_A
        M_i = m_i[-1]
        MA_i = m_a[-1]
        MB_i = m_b[-1]
        
        R_values.append(R_i)
        M_values.append(M_i)
        MA_values.append(MA_i)
        MB_values.append(MB_i)
        
        cont += 1
        print(cont)
    
    return (np.array(R_values), np.array(M_values), np.array(MA_values), np.array(MB_values))

###############################################################################
# TEST
###############################################################################

# Read the data of soft_EoS
eos_data = pd.read_excel("eos_soft.xlsx")
rho_data = eos_data['Density'].values
p_data = eos_data['Pressure'].values



def find_lambda (pc, l_target):
    """
    Finds the alpha parameter for which lambda becomes lambda_target by using the secant method for finding the roots of a function.

    Parameters
    ----------
    pc : float
        Value of central pressure of fluid A.
    l_target : float
        Target value of lambda.

    Raises
    ------
    ValueError
        Raises an error when the root finding algorithm does not converge.

    Returns
    -------
    TYPE
        Value of alpha for which lambda satisfies the target condition.

    """
    def f(alpha):
        """
        Auxiliary function used to calculate the difference of lambda to a guiven lambda_target, as a function of alpha.
        The rest of the parameters (pc) are set as guiven values.

        Parameters
        ----------
        alpha : float
            pc_B/p_A.

        Returns
        -------
        float
            differnce betwen lambda (M_B/M) and the target lambda.

        """
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), (1e-6, 50), 1e-3)
        l = m_B[-1]/m[-1]
        return l-l_target
    a_guess = l_target
    result = opt.root_scalar(f, x0=a_guess, method='secant', x1=a_guess*1.1)
    if result.converged:
        return result.root
    else:
        raise ValueError("Root-finding did not converge")
        
pc = 1e-5
l_target = 0.1
#a_guess = 0.0

alpha = find_lambda(pc, l_target)

print(f"El valor de alpha para lamba={l_target} es: {alpha}")