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

def eos_B (p_B): # DM
    """
    Polytropic EOS for the star B. The polytropic constant and exponent are defined in the unit sections. 
    
    Parameters:
        p_B (float): Preassure of the fluid B.
        
    Returns:
        float: Density of the fluid B (rho) at preassure p_B.
    """

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
            print("The star has a Halo of Dark Matter") #DEBUG
            
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
        print("The star has a core of Dark Matter") #DEBUG
        
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

def get_inputs():
    """
    Prompts the user for input parameters to configure the DANTE solver.
    
    Raises
    ------
    ValueError
        If the user provides an invalid input that does not match the expected options.
    
    Returns
    -------
    mode : int or str
        0 for calculating data, 1 for plotting data, or "DEBUG" mode.
    data_type : int
        1 for TOV of fluid A, 2 for TOV of fluid B, 3 for both fluids, 4 for MR curves.
    eos_choice : str
        'soft', 'mid', or 'stiff' for the equation of state.
    param_choice : str or None
        'a' for Alpha, 'l' for Lambda, or None if not applicable.
    param_value : float or None
        The value of the chosen DM parameter or None if not applicable.
    central_pressure : float or None
        The central pressure value or None if not applicable.
    """
    while True:
        try:
            mode = int(input("What do you want to do? Calculate new data (0) or plot existing data (1): "))
            if mode not in [0, 1]:
                mode = "DEBUG"
            break
        except ValueError:
            mode = "DEBUG"
            break
    
    if mode != "DEBUG":
        while True:
            try:
                data_type = int(input("What type of data?\n1: Solve the TOV equation for a NS\n2: Solve the TOV equation for a DM star\n3: Solve the TOV equation for a DANS\n4: MR curves\nEnter choice (1-4): "))
                if data_type not in [1, 2, 3, 4]:
                    raise ValueError("Invalid choice. Enter a number between 1 and 4.")
                break
            except ValueError as e:
                print(e)
        
        if data_type != 2:
            while True:
                try:
                    eos_choice = int(input("What EoS do you want to use?\n1: Soft\n2: Middle\n3: Stiff\nEnter choice (1-3): "))
                    eos_options = {1: 'soft', 2: 'mid', 3: 'stiff'}
                    if eos_choice not in eos_options:
                        raise ValueError("Invalid choice. Enter 1 for Soft, 2 for Mid, or 3 for Stiff.")
                    eos_choice = eos_options[eos_choice]
                    break
                except ValueError as e:
                    print(e)
        else:
            eos_choice = None
        
        if data_type in [3, 4]:
            while True:
                try:
                    param_choice = input("What parameter of DM do you want? Alpha (a) or Lambda (l): ").strip().lower()
                    if param_choice not in ['a', 'l']:
                        raise ValueError("Invalid choice. Enter 'a' for Alpha or 'l' for Lambda.")
                    param_value = float(input(f"Enter the value of {param_choice.upper()}: "))
                    break
                except ValueError as e:
                    print(e)
        else:
            param_choice, param_value = None, None
        
        if data_type != 4:
            while True:
                try:
                    central_pressure = float(input("What is the central pressure of the fluid? "))
                    if central_pressure <= 0:
                        raise ValueError("Central pressure must be a positive number.")
                    break
                except ValueError as e:
                    print(e)
        else:
            central_pressure = None
                
    else:
        print("Using DEBUG data")
        mode, data_type, eos_choice, param_choice, param_value, central_pressure = (0, 1, 'soft', None, None, 3e-5)
        
    return mode, data_type, eos_choice, param_choice, param_value, central_pressure

###############################################################################
# Define the parameters
###############################################################################

print("Welcome to DANTE: the Dark-matter Admixed Neutron-sTar solvEr.")

mode, d_type, eos_c, param_c, param_val, p_c = get_inputs()

print("\nUser Inputs:", mode, d_type, eos_c, param_c, param_val, p_c)

###############################################################################
# Create the data
###############################################################################
if mode == 0:
    
    if d_type == 1: # Solve the TOV for fluid A
        eos_data = pd.read_excel(f"data_eos/eos_{eos_c}.xlsx")
        rho_data = eos_data['Density'].values
        p_data = eos_data['Pressure'].values
        data = {}
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0,p_c,0,0,0), (1e-6, 100), 1e-3)
        data["r"] = r
        data["m"] = m
        data["p_A"] = p_A
        data["p_B"] = p_B
        data["m_A"] = m_A
        data["m_B"] = m_B
        data["R_A"] = R_A
        df = pd.DataFrame(data)
        df.to_csv(f"data/{d_type}_{eos_c}_{p_c}", index=False)
        print("Data saved")




