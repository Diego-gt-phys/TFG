# -*- coding: utf-8 -*-
"""
Version: 2.0

Created on Thu Mar 27 09:54:31 2025

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

def eos_B (p_B): # DM
    """
    Polytropic EOS for the star B. The polytropic constant and exponent are defined in the unit sections. 
    
    Parameters:
        p_B (float): Preassure of the fluid B.
        
    Returns:
        float: Density of the fluid B (rho) at preassure p_B.
    """
    # We work with an Ideal Fermi Gass
    Gamma = 5/3 # Polytropic coeficient for a degenetare (T=0) IFG
    K = ((dm_m)**(-8/3))*8.0165485819726 # Polytropic constant for a degenetare (T=0) IFG

    if p_B <= 0:
        return 0  # Avoid invalid values
        
    rho = (p_B / K) ** (1/Gamma)
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
    dmA_dr : float
        rate of change of the A mass.
    dmB_dr : float
        rate of change of the B mass.
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
    
    print("TOV solved")

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

def MR_curve_lambda (pc_range, l, r_range, h, n):
    """
    Creates the mass radius curve of a family of 2-fluid stars by solving the TOV equations.
    The amount of matter B inside the star is fixed for every point by the value of l
    
    Parameters
    ----------
    pc_range : tuple
        Range of central pressures to integrate: (pc_start, pc_end)
    l : float
        Fraction of the total mass of B inside the star: l = M_B/M
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
        alpha = find_lambda(pc, l)
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
    if l_target == 0:
        return 0
    else:
        a_guess = l_target
        result = opt.root_scalar(f, x0=a_guess, method='secant', x1=a_guess*1.1)
        if result.converged:
            return result.root
        else:
            raise ValueError("Root-finding did not converge")

def find_sc (M_target, l_target):
    """
    Function that calculates the starting conditions [p_c, alpha], that guive a certain M_target, l_target.
    The function uses a hybrid method to solve the roots of a vector function
    
    Parameters
    ----------
    M_target : float
        Desidered mass of the star.
    l_target : float
        Desidered fraction of mass correspondant to the DM.
    
    Raises
    ------
    ValueError
        When the numerical method to solve the roots does not converge.
        
    Returns
    -------
    float
        Central pressure of fluid A.
    float
        Alpha parameter (p_c_B/p_c_A).
    """
    def f(x):
        """
        Auxilary function used to calculate the residuals of M and alpha to their respective target values.
        This function takes a vector x=[pc, alpha] and calculates [M,lambda]
        
        Parameters
        ----------
        x : list
            vector input [p_c, alpha].
            
        Returns
        -------
        M-M_target : float
            Residual of calculated M to a M_target.
        l-l_target : float
            Residual of calculated l to a l_taget.
        """
        p_c, alpha = x
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, p_c, alpha*p_c, 0, 0), (1e-6, 100), 1e-3)
        M = m[-1]
        l = m_B[-1]/m[-1]
        
        return M-M_target, l-l_target
    
    x0 = [M_target*1e-4, l_target] # Inital Guess
    sol = opt.root(f, x0, method='hybr')
    
    if sol.success:
        return sol.x  # Returns (pc, alpha)
    else:
        raise ValueError(f"->{sol.message}")

def get_inputs (mode_DB, s_type_DB, d_type_DB, eos_c_DB, dm_m_DB, p1_c_DB, p1_v_DB, p2_c_DB, p2_v_DB):
    """
    Promts the user for inputs parameters to configure the DANTE.
    If in the users first input is incorrect then enters DEBUG mode.
    DEBUG mode skips the questions uses the parameters for the return.

    Parameters
    ----------
    mode_DB : int or str
        0 for calculating data, 1 for plotting data.
    s_type_DB : int
        1 for a Neutron Star, 2 for a Dark Matter Star, 3 for a DANS.
    d_type_DB : int
        0 for a TOV solution or 1 for a MR curve.
    eos_c_DB : str
        'soft', 'middle', 'stiff' for the EoS to use for baryonic matter.
    dm_m_DB : float or None
        Mass of the Dark Matter particle in GeV.
    p1_c_DB : str or None
        'p_c' for central pressure, 'M' for total mass.
    p1_v_DB : float or None
        Value of parameter 1. Units are solar masses and kilometers
    p2_c_DB : str or None
        'a' for the central pressure of DM fraction or 'l' for the DM fraction of central mass.
    p2_v_DB : float or None
        Value of parameter 2.

    Returns
    -------
    Same definitons of parameters
    """
    while True: # Find mode
        try:
            mode = int(input("\nWhat do you want to do? \n0 : Calculate new data.\n1 : Plot existing data. \nEnter choice (0, 1) -> "))
            if mode not in [0, 1]:
                mode = "DEBUG"
            break
            
        except ValueError:
            mode = "DEBUG"
            break
            
    if mode != "DEBUG": # Use manual mode
        while True: # Find s_type
            try:
                s_type = int(input("\nWith what system do you want to work with? \n1 : Neutron Star. \n2 : Dark Matter Star. \n3 : DANS. \nEnter choice (1, 2, 3) -> "))
                if s_type not in [1, 2, 3]:
                    raise ValueError("Invalid choice. Enter a number between 1 and 3.")
                break
            except ValueError as e:
                print(e)
        
        while True: # Find d_type
            try:
                d_type = int(input("\nWhich type of calculation? \n0 : TOV equation. \n1 : Mass Radius curve. \n Enter choice (0, 1) -> "))
                if d_type not in [0, 1]:
                    raise ValueError("Invalid choice. Enter a 0 or 1.")
                break
            except ValueError as e:
                print(e)
        
        if s_type != 2: # Find eos_c
            while True:
                try:
                    eos_c = int(input("\nWhat EoS do you want to use?\n1: Soft\n2: Middle\n3: Stiff\nEnter choice (1-3) -> "))
                    eos_opt = {1:'soft', 2:'middle', 3:'stiff'}
                    if eos_c not in eos_opt:
                        raise ValueError("Invalid choice. Enter 1 for Soft, 2 for Mid, or 3 for Stiff.")
                    eos_c = eos_opt[eos_c]
                    break
                except ValueError as e:
                    print(e)
            
        else:
            eos_c = 'soft'
        
        if s_type != 1: # Find dm_m
            while True:
                try:
                    dm_m = float(input("\nWhat is the mass of the DM particle in GeV? \n -> "))
                    break
                except ValueError as e:
                    print(e)
        else:
            dm_m = 1
        
        if d_type  == 0: # Calculate the parameters when it applies 
            while True: # Calculate p1
                try:
                    p1_c = input("\nWhat parameter of matter do you want to use? \n'pc' : Central Pressure. \n'M' : Total mass of the star. \nEnter choice ('pc','M') -> ")
                    if p1_c not in ["pc", "M"]:
                        raise ValueError("Invalid choice. Enter 'pc' for central pressure or 'M' for total mass.")
                    p1_v = float(input(f"\nEnter the value of {p1_c} -> "))
                    break
                except ValueError as e:
                    print(e)
            
            if s_type == 3: # Calculate p2
                while True:
                    try:
                        p2_c = input("\nWhat parameter of DM do you want to use? \n'a' : central pressure fraction. \n'l' : Mass fraction of DM. \nEnter choice ('a', 'l') -> ")
                        if p2_c not in ["a", "l"]:
                            raise ValueError("Invalid choice. Enter 'a' for Alpha or 'l' for Lambda.")
                        p2_v = float(input(f"\nEnter the value of {p2_c} -> "))
                        break
                    except ValueError as e:
                        print(e)
            else:
                p2_c = p2_v = None
                
        else:
            p1_c = p1_v = p2_c = p2_v = None
    
    else: # Use DEBUG mode
        print("\nUsing DEBUG data:")
        mode, s_type, d_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v = mode_DB, s_type_DB, d_type_DB, eos_c_DB, dm_m_DB, p1_c_DB, p1_v_DB, p2_c_DB, p2_v_DB
    
    return mode, s_type, d_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v


