# -*- coding: utf-8 -*-
"""
Version: 2.3

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
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as opt
import multiprocessing as mp
from tqdm import tqdm

# Physical parameters (solar mass = 1.98847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses
PCS = {"soft": (2.785e-6, 5.975e-4), "middle": (2.747e-6, 5.713e-4), "stiff": (2.144e-6, 2.802e-4)} # Central pressure intervals for the MR curves 


###############################################################################
# Define the functions
###############################################################################

def eos_A (p_A, p_data, rho_data): # BM
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

def eos_B (p_B, p_data_dm, rho_data_dm): # DM
    """
    Guiven the arrays 'p_data_dm' and 'rho_data_d,' which contain the information for the equation of state for the dark matter,
    this funtion interpolates the value of rho for a guiven  p. 

    Parameters
    ----------
    p_B : float
        Preassure of fluid B at which we want to evaluate the eos.
        
    Returns
    -------
    rho : float
        Density associated to the preassure guiven.
    """
    if p_B <= 0:
        return 0
    
    elif p_B <= 1e-20:
        Gamma = 5/3
        K = ((dm_m)**(-8/3))*8.0165485819726
        rho = (p_B / K) ** (1/Gamma)
    
    else:
        interp_func = interp1d(p_data_dm, rho_data_dm, kind='linear', fill_value='extrapolate')
        rho = interp_func(p_B)
    
    return rho

def system_of_ODE (r, y, p_data, rho_data, p_data_dm, rho_data_dm):
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
    rho_A = eos_A(p_A, p_data, rho_data)
    rho_B = eos_B(p_B, p_data_dm, rho_data_dm)
    
    dm_dr = 4 * np.pi * (rho_A + rho_B) * r**2
    
    dmA_dr = 4 * np.pi * (rho_A) * r**2
    
    dmB_dr = 4 * np.pi * (rho_B) * r**2
    
    dphi_dr = (G * m + 4 * np.pi * G * r**3 * (p_A + p_B)) / (r * (r - 2 * G * m))
    
    dpA_dr = -(rho_A + p_A) * dphi_dr
    
    dpB_dr = -(rho_B + p_B) * dphi_dr
    
    return (dm_dr, dpA_dr, dpB_dr, dmA_dr, dmB_dr)

def RK4O_with_stop (y0, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm):
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
        k1 = h * np.array(system_of_ODE(r, y, p_data, rho_data, p_data_dm, rho_data_dm))
        k2 = h * np.array(system_of_ODE(r + h / 2, y + k1 / 2, p_data, rho_data, p_data_dm, rho_data_dm))
        k3 = h * np.array(system_of_ODE(r + h / 2, y + k2 / 2, p_data, rho_data, p_data_dm, rho_data_dm))
        k4 = h * np.array(system_of_ODE(r + h, y + k3, p_data, rho_data, p_data_dm, rho_data_dm))

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

def TOV_solver (y0, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm):
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
    
    r_values, y_values, R_A = RK4O_with_stop(y0, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm)
    
    m_values = y_values[:, 0]
    p_A_values = y_values[:, 1]
    p_B_values = y_values[:, 2]
    m_A_values = y_values[:, 3]
    m_B_values = y_values[:, 4]
    
    print("Star Solved")
    #print(p_A_values[0], p_B_values[0]/p_A_values[0]) #DEBUG
    
    return (r_values, m_values, p_A_values, p_B_values, m_A_values, m_B_values, R_A)

def solve_single_star_MR_curve(args):
    pc, alpha, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm = args
    r_i, m_i, p_A, p_B, m_a, m_b, R_A = TOV_solver((0, pc, alpha * pc, 0, 0), r_range, h, p_data, rho_data, p_data_dm, rho_data_dm)
    return R_A, m_i[-1], m_a[-1], m_b[-1]

def MR_curve (pc_range, alpha, r_range, h, n, p_data, rho_data, p_data_dm, rho_data_dm):
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
    args_list = [(pc, alpha, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm) for pc in pc_list]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(solve_single_star_MR_curve, args_list), total=n, desc="Processing MR_curve_lambda"))

    R_values, M_values, MA_values, MB_values = zip(*results)
    return np.array(R_values), np.array(M_values), np.array(MA_values), np.array(MB_values)

def solve_single_star_MR_curve_lambda(args):
    pc, l, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm = args
    alpha = find_alpha(pc, l, p_data, rho_data, p_data_dm, rho_data_dm)
    r_i, m_i, p_A, p_B, m_a, m_b, R_A = TOV_solver((0, pc, alpha * pc, 0, 0), r_range, h, p_data, rho_data, p_data_dm, rho_data_dm)
    return R_A, m_i[-1], m_a[-1], m_b[-1]

def MR_curve_lambda (pc_range, l, r_range, h, n, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Creates the mass radius curve of a family of 2-fluid stars by solving the TOV equations.
    The amount of matter B inside the star is fixed for every point by the value of l.
    
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
    args_list = [(pc, l, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm) for pc in pc_list]

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(solve_single_star_MR_curve_lambda, args_list), total=n, desc="Processing MR_curve_lambda"))

    R_values, M_values, MA_values, MB_values = zip(*results)
    return np.array(R_values), np.array(M_values), np.array(MA_values), np.array(MB_values)

def find_alpha (pc, l_target, p_data, rho_data, p_data_dm, rho_data_dm):
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
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), (1e-6, 50), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
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

def find_pc (M_target, s_type, alpha, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Finds the central pressure that guives a star a certain Target Mass by using the secant method for finding the roots of a function.

    Parameters
    ----------
    M_target : float
        Target Mass.
    s_type : int
        0 for a NS, or 1 for a DMS.
    alpha : float or None
        If DANS, alpha parameter.

    Raises
    ------
    ValueError
        When root finding algorithm does not converge.

    Returns
    -------
    float
        Central pressure that guives target mass.
    """
    def f(pc):
        """
        Auxiliaryfunction that guives the residuals of a star with central pressure pc.

        Parameters
        ----------
        pc : float
            cental pressure.

        Returns
        -------
        float
            Residual of Mass compared to M_target.
        """
        if s_type == 1:
            r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, 0, 0, 0), (1e-6, 50), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
        elif s_type == 2:
            r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, 0, pc, 0, 0), (1e-6, 50), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
        else:
            r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), (1e-6, 50), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
        M = m[-1]
        return M - M_target
    pc_guess = M_target*5e-5
    result = opt.root_scalar(f, x0=pc_guess, method='secant', x1=pc_guess*1.1)
    if result.converged:
        return result.root
    else:
        raise ValueError("Root-finding did not converge")

def find_sc (M_target, l_target, p_data, rho_data, p_data_dm, rho_data_dm):
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
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((1, p_c, alpha*p_c, 0, 0), (1e-6, 100), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
        M = m[-1]
        l = m_B[-1]/m[-1]
        #print(M, l) #DEBUG
        
        return M-M_target, l-l_target
    
    x0 = [M_target*5e-5, l_target] # Inital Guess
    sol = opt.root(f, x0, method='hybr')
    
    if sol.success:
        return sol.x  # Returns (pc, alpha)
    else:
        raise ValueError(f"->{sol.message}")

def read_eos (eos_c):
    """
    Reads an equation of state (EOS) file in Excel format and extracts density 
    and pressure data.

    Parameters
    ----------
    eos_c : str
        Identifier of the EOS file to be read. The function expects a file 
        named 'eos_{eos_c}.xlsx' located in the 'data_eos' directory.

    Returns
    -------
    p_data : numpy.ndarray
        Array containing pressure values extracted from the EOS file.
    rho_data : numpy.ndarray
        Array containing density values extracted from the EOS file.
    """
    eos_data = pd.read_excel(f"data_eos\eos_{eos_c}.xlsx")
    rho_data = eos_data['Density'].values
    p_data = eos_data['Pressure'].values
    return p_data, rho_data

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
        0 for a TOV solution, 1 for a MR curve, 2 for a l_max finder.
    eos_c_DB : str
        'soft', 'middle', 'stiff' for the EoS to use for baryonic matter.
    dm_m_DB : float
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
                if s_type==3:
                    d_type = int(input("\nWhich type of calculation? \n0 : TOV equation. \n1 : Mass Radius curve. \n2 : Find l max. \n Enter choice (0, 1, 2) -> "))
                    if d_type not in [0, 1, 2]:
                        raise ValueError("Invalid choice. Enter a 0, 1 or 2.")
                    break
                else:
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
        
        if d_type  in [0, 2]: # Calculate the parameters when it applies 
            while True: # Calculate p1
                try:
                    p1_c = input("\nWhat parameter of matter do you want to use? \n'pc' : Central Pressure. \n'M' : Total mass of the star. \nEnter choice ('pc','M') -> ")
                    if p1_c not in ["pc", "M"]:
                        raise ValueError("Invalid choice. Enter 'pc' for central pressure or 'M' for total mass.")
                    p1_v = float(input(f"\nEnter the value of {p1_c} -> "))
                    break
                except ValueError as e:
                    print(e)
            
            if s_type == 3 and d_type != 2: # Calculate p2
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
            p1_c = p1_v = None
            if s_type == 3:
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
    
    else: # Use DEBUG mode
        print("\nUsing DEBUG data:")
        mode, s_type, d_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v = mode_DB, s_type_DB, d_type_DB, eos_c_DB, dm_m_DB, p1_c_DB, p1_v_DB, p2_c_DB, p2_v_DB
    
    return mode, s_type, d_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v

def Save_TOV (s_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Calculates and Saves the solution of the TOV, for a systems whose imputs are 
    this functions parameters.
    
    Parameters
    ----------
    s_type : int
        Type of star configuration:
        - 1: Neutron Star (NS)
        - 2: Dark Matter Admixed Neutron Star (DANS)
        - 3: Double Admixed Neutron Star (DANS with extra parameters)
    eos_c : str
        Identifier of the equation of state (EOS) used for baryonic matter.
    dm_m : float
        Dark matter particle mass, relevant for s_type = 2 or 3.
    p1_c : str
        Parameter type for the first input value (e.g., "M" for mass, "pc" for central pressure).
    p1_v : float
        Value corresponding to the first parameter.
    p2_c : str
        Parameter type for the second input value (only used for s_type = 3).
    p2_v : float
        Value corresponding to the second parameter (only used for s_type = 3).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the computed radial profile of the star, including:
        - r : Radial coordinate values.
        - m : Enclosed mass at each radius.
        - p_A : Pressure of baryonic matter.
        - p_B : Pressure of dark matter.
        - m_A : Mass contribution from baryonic matter.
        - m_B : Mass contribution from dark matter.
        - R_A : Outer radius of the baryonic component.

    Notes
    -----
    - The function uses interpolation for the EOS and integrates the TOV 
      equations using a Runge-Kutta method.
    - The output filename is constructed based on the input parameters.
    """
    if s_type == 3: # DANS
        if p1_c == "M" and p2_c == "l":
            pc, alpha = find_sc(p1_v, p2_v, p_data, rho_data, p_data_dm, rho_data_dm)
        elif p1_c == "M" and p2_c == "a": #DEBUG
            pc = find_pc(p1_v, s_type, p2_v, p_data, rho_data, p_data_dm, rho_data_dm)
            alpha = p2_v
        elif p2_c == "l" and p1_c == "pc": #DEBUG
            pc = p1_v
            alpha = find_alpha(p1_v, p2_v, p_data, rho_data, p_data_dm, rho_data_dm)
            print(alpha)
        else:
            pc, alpha = [p1_v, p2_v]
        
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), (1e-6, 100), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
        
    else:
        if p1_c == "M":
            pc = find_pc(p1_v, s_type, None, p_data, rho_data, p_data_dm, rho_data_dm)
        else:
            pc = p1_v
            
        if s_type == 1:
            r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, 0, 0, 0), (1e-6, 100), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
        else:
            r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, 0, pc, 0, 0), (1e-6, 100), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
            
    data = {}
    data["r"] = r
    data["m"] = m
    data["p_A"] = p_A
    data["p_B"] = p_B
    data["m_A"] = m_A
    data["m_B"] = m_B
    data["R_A"] = R_A
    df = pd.DataFrame(data)
    
    if s_type == 1:
        df.to_csv(f"data\{s_type}_{d_type}_{eos_c}_{p1_c}_{p1_v}.csv", index=False)
    elif s_type == 2:
        df.to_csv(f"data\{s_type}_{d_type}_{dm_m}_{p1_c}_{p1_v}.csv", index=False)
    else:
        df.to_csv(f"data\{s_type}_{d_type}_{eos_c}_{dm_m}_{p1_c}_{p1_v}_{p2_c}_{p2_v}.csv", index=False)
    
    return df

def solve_single_star_MR_curve_1f(args):
    
    pc, s_type, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm = args
        
    if s_type == 1:
        r_i, m_i, p_A, p_B, m_a, m_b, R_A = TOV_solver((0, pc, 0, 0, 0), r_range, h, p_data, rho_data, p_data_dm, rho_data_dm)
    elif s_type == 2:
        r_i, m_i, p_A, p_B, m_a, m_b, R_A = TOV_solver((0, 0, pc, 0, 0), r_range, h, p_data, rho_data, p_data_dm, rho_data_dm)
            
    R_i = r_i[-1]
    M_i = m_i[-1]
        
    return R_i, M_i

def MR_curve_1f(pc_range, s_type, r_range, h, n, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Creates the mass-radius curve of a family of 1-fluid stars by solving the TOV equations.
    The different masses are calculated by changing the central pressure. Now parallelized.
    
    Parameters
    ----------
    pc_range : tuple
        Range of central pressures to integrate: (pc_start, pc_end)
    s_type : int
        Type of star: 1 for NS or 2 for DMS
    r_range : tuple
        Range of radius integration: (r_0, r_max)
    h : float
        Integration step.
    n : int
        Number of datapoints of the curve.
        
    Returns
    -------
    R_values : array
        Array containing the radii of the stars.
    M_values : array
        Array containing the total masses of the stars.
    """    
    pc_start, pc_end = pc_range
    pc_list = np.geomspace(pc_start, pc_end, n)
    args_list = [(pc, s_type, r_range, h, p_data, rho_data, p_data_dm, rho_data_dm) for pc in pc_list]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(solve_single_star_MR_curve_1f, args_list), total=n, desc="Processing Stars"))
        
    R_values, M_values = zip(*results)
    return np.array(R_values), np.array(M_values)

def Save_MR (s_type, eos_c, dm_m, p2_c, p2_v, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Calculates and Saves the solution of the TOV, for a systems whose imputs are 
    this functions parameters.
    
    Parameters
    ----------
    s_type : int
        Type of star configuration:
        - 1: Neutron Star (NS)
        - 2: Dark Matter Admixed Neutron Star (DANS)
        - 3: Double Admixed Neutron Star (DANS with extra parameters)
    eos_c : str
        Identifier of the equation of state (EOS) used for baryonic matter.
    dm_m : float
        Dark matter particle mass, relevant for s_type = 2 or 3.
    p2_c : str
        Parameter type for the second input value (only used for s_type = 3).
    p2_v : float
        Value corresponding to the second parameter (only used for s_type = 3).
        
    Returns
    -------
    df : Pandas DataFrame
        DF containing all the datapoints of the MR curve.
    """
    data = {}
    pc_range = PCS[eos_c]
    
    if s_type != 3:
        R, M = MR_curve_1f(pc_range, s_type, (1e-6, 100), 1e-3, 30, p_data, rho_data, p_data_dm, rho_data_dm)
        data["R"] = R
        data["M"] = M
        df = pd.DataFrame(data)
        if s_type == 1:
            df.to_csv(f"data\{s_type}_{d_type}_{eos_c}.csv", index=False)
        else:
            df.to_csv(f"data\{s_type}_{d_type}_{dm_m}.csv", index=False)
            
    else:
        if p2_c == 'a':
            R, M, M_A, M_B = MR_curve(pc_range, p2_v, (1e-6, 100), 1e-3, 30,  p_data, rho_data, p_data_dm, rho_data_dm)
        else:
            R, M, M_A, M_B = MR_curve_lambda(pc_range, p2_v, (1e-6, 100), 1e-3, 30,  p_data, rho_data, p_data_dm, rho_data_dm)
        
        data["R"] = R
        data["M"] = M
        data["M_A"] = M_A
        data["M_B"] = M_B
        df = pd.DataFrame(data)
        df.to_csv(f"data\{s_type}_{d_type}_{eos_c}_{dm_m}_{p2_c}_{p2_v}.csv", index=False)
        
    return df

def calc_CDIFG (dm_m):
    """
    Function that calculates the numerical values of the Relativistic Equation of State for a
    Completley Degenerate, Ideal Fermi Gas of particle mass = dm_m.
    
    Due to numerical issues for values of P lower than 10^-12 it uses the 21st order taylor expansion.
    
    Parameters
    ----------
    dm_m : float
        Mass of the particle mass in GeV.
        
    Returns
    -------
    Pandas DataFrame
        DF containing all the datapoints of EoS.
    """
    def phi(x):
        return np.sqrt(1 + x**2) * ((2/3) * x**3 - x) + np.log(x + np.sqrt(1 + x**2))
    
    def psi(x):
        return np.sqrt(1 + x**2) * (2 * x**3 + x) - np.log(x + np.sqrt(1 + x**2))
        
    def phi_21st(x):
        return (8/15) * x**5 - (4/21) * x**7 + (1/9) * x**9 - (5/66) * x**11 + (35/624) * x**13 - (7/160) * x**15 + (77/2176) * x**17 - (143/4864) * x**19 + (715/28672) * x**21
    
    def psi_21st(x):
        return (8/3) * x**3 + (4/5) * x**5 - (1/7) * x**7 + (1/18) * x**9 - (5/176) * x**11 + (7/416) * x**13 - (7/640) * x**15 + (33/4352) * x**17 - (429/77824) * x**19 + (715/172032) * x**21
    
    k = (dm_m)**(4) * 1.4777498161008e-3
        
    x_vals = np.geomspace(1e-6, 10, 500)
    
    pressures = []
    densities = []
    
    for x in x_vals:
        P_analytical = k * phi(x)
        
        if P_analytical >= 1e-12:
            P = P_analytical
            rho = k * psi(x)
        else:
            P = k * phi_21st(x)
            rho = k * psi_21st(x)
            
        pressures.append(P)
        densities.append(rho)
        
    df = pd.DataFrame({'rho': densities,'P': pressures})
    
    df.to_excel(rf'data_eos\eos_cdifg_{dm_m}.xlsx', index=False)
        
    return df

def read_create_dm_eos (dm_m):
    """
    Reads an equation of state (EOS) file in Excel format and extracts density 
    and pressure data of the dark matter.
    
    Parameters
    ----------
    dm_m : float
        mass of the dark matter particle in GeV.
        
    Returns
    -------
    p_data_dm : numpy.ndarray
        Array containing pressure values extracted from the EOS file.
    rho_data_dm : numpy.ndarray
        Array containing density values extracted from the EOS file.
    """
    try:
        eos_data = pd.read_excel(f'data_eos\eos_cdifg_{dm_m}.xlsx')
        rho_data_dm = eos_data['rho'].values
        p_data_dm = eos_data['P'].values
    except FileNotFoundError:
        eos_data = calc_CDIFG(dm_m)
        rho_data_dm = eos_data['rho'].values
        p_data_dm = eos_data['P'].values
        
    return p_data_dm, rho_data_dm

def find_alpha_max(p1_c, p1_v, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Finds the alpha parameter for which lambda (M_B/M) reaches its maximum value,
    using scalar optimization.
    
    Parameters
    ----------
    p1_c : str
        Parameter type for the first input value (e.g., "M" for mass, "pc" for central pressure).
    p1_v : float
        Value corresponding to the first parameter.
    p_data : array_like
        Pressure data for fluid A (baryonic matter).
    rho_data : array_like
        Density data for fluid A (baryonic matter).
    p_data_dm : array_like
        Pressure data for fluid B (dark matter).
    rho_data_dm : array_like
        Density data for fluid B (dark matter).
        
    Returns
    -------
    float
        Value of alpha for which lambda is maximized.
    """
    
    def lambda_func(alpha):
        """
        Auxiliary function used to calculate lambda = M_B / M as a function of alpha.
        
        Parameters
        ----------
        alpha : float
            Ratio of central pressure of fluid B to fluid A (pc_B / pc_A).
            
        Returns
        -------
        float
            The value of lambda for the given alpha.
        """
        if p1_c == 'M':
            pc = find_pc(p1_v, s_type, alpha, p_data, rho_data, p_data_dm, rho_data_dm)
            
        else:
            pc = p1_v
        
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver(
            (0, pc, alpha * pc, 0, 0),
            (1e-6, 50),
            1e-3,
            p_data, rho_data, p_data_dm, rho_data_dm
        )
        print(m_B[-1] / m[-1]) #DEBUG
        return m_B[-1] / m[-1]
    
    def neg_lambda(alpha):
        """
        Negative lambda, used to find the maximum by minimizing this value.
        """
        return -lambda_func(alpha)
    
    result = opt.minimize_scalar(neg_lambda, bounds=(1, 6), method='bounded')
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization did not converge")

def save_l_max(s_type, eos_c, dm_m, p1_c, p1_v, p_data, rho_data, p_data_dm, rho_data_dm):
    """
    Calculates and Saves the solution of the TOV for a star with maximum amount of DM.
    
    Parameters
    ----------
    s_type : int
        Type of star configuration:
        - 1: Neutron Star (NS)
        - 2: Dark Matter Admixed Neutron Star (DANS)
        - 3: Double Admixed Neutron Star (DANS with extra parameters)
    eos_c : str
        Identifier of the equation of state (EOS) used for baryonic matter.
    dm_m : float
        Dark matter particle mass, relevant for s_type = 2 or 3.
    p1_c : str
        Parameter type for the first input value (e.g., "M" for mass, "pc" for central pressure).
    p1_v : float
        Value corresponding to the first parameter.
        
    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the computed radial profile of the star.
    """
    alpha = find_alpha_max(p1_c, p1_v, p_data, rho_data, p_data_dm, rho_data_dm)
    
    if p1_c == 'M':
        pc = find_pc(p1_v, s_type, alpha, p_data, rho_data, p_data_dm, rho_data_dm)
        
    else:
        pc = p1_v
        
    r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), (1e-6, 100), 1e-3, p_data, rho_data, p_data_dm, rho_data_dm)
    data = {}
    data["r"] = r
    data["m"] = m
    data["p_A"] = p_A
    data["p_B"] = p_B
    data["m_A"] = m_A
    data["m_B"] = m_B
    data["R_A"] = R_A
    df = pd.DataFrame(data)
    df.to_csv(f"data\{s_type}_{d_type}_{eos_c}_{dm_m}_{p1_c}_{p1_v}.csv", index=False)
    
    return df

if __name__ == '__main__':
###############################################################################
# Define the parameters
###############################################################################
    
    print("Welcome to DANTE: the Dark-matter Admixed Neutron-sTar solvEr.")
    
    mode, s_type, d_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v = get_inputs(1, 3, 2, 'stiff', 0.9, 'M', 1.5, 'None', None)
    
    print(f"\nUser Inputs: {mode}, {s_type}, {d_type}, '{eos_c}', {dm_m}, '{p1_c}', {p1_v}, '{p2_c}', {p2_v}\n")
    
###############################################################################
# Calculate the data
###############################################################################
    
    if mode == 0:
        p_data, rho_data = read_eos(eos_c)
        p_data_dm, rho_data_dm = read_create_dm_eos(dm_m)
                
        if d_type == 0:
            df = Save_TOV(s_type, eos_c, dm_m, p1_c, p1_v, p2_c, p2_v, p_data, rho_data, p_data_dm, rho_data_dm)
            print()
            print(df)
            print("\nData Saved.")
        
        elif d_type == 1:
            df = Save_MR(s_type, eos_c, dm_m, p2_c, p2_v, p_data, rho_data, p_data_dm, rho_data_dm)
            print()
            print(df)
            print("\nData Saved.")
            
        elif d_type == 2:
            df = save_l_max(s_type, eos_c, dm_m, p1_c, p1_v, p_data, rho_data, p_data_dm, rho_data_dm)
            print()
            print(df)
            print("\nData Saved.")
    
###############################################################################
# Plot the data
###############################################################################
    
    if mode == 1:
        
        if d_type == 0:
            # Read data
            if s_type == 1:
                df = pd.read_csv(f"data\{s_type}_{d_type}_{eos_c}_{p1_c}_{p1_v}.csv")
            elif s_type == 2:
                df = pd.read_csv(f"data\{s_type}_{d_type}_{dm_m}_{p1_c}_{p1_v}.csv")
            else:
                df = pd.read_csv(f"data\{s_type}_{d_type}_{eos_c}_{dm_m}_{p1_c}_{p1_v}_{p2_c}_{p2_v}.csv")
            
            r = df['r']
            m = df['m']
            p_A = df['p_A']
            p_B = df['p_B']
            m_A = df['m_A']
            m_B = df['m_B']
            
            # Configure the plot
            fig, ax1 = plt.subplots(figsize=(9.71, 6))
            colors = sns.color_palette("Set1", 10)
            eos_colors = {"soft": 0, "middle": 1, "stiff": 2, "DM": 3}
            c = eos_colors[eos_c]
            
            # Plot pressures 
            if s_type != 2:
                pa = ax1.plot(r, p_A, label=rf'$p_{{{eos_c}}}$', color = colors[c], linewidth=1.5, linestyle='-')
            if s_type != 1:
                pb = ax1.plot(r, p_B, label=r'$p_{{DM}}$', color = colors[3], linewidth=1.5, linestyle='-')
            ax1.set_xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
            ax1.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
            ax1.tick_params(axis='y', colors='k')
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            #ax1.set_yscale('log')
            
            # Plot Mass
            ax2 = ax1.twinx()
            m = ax2.plot(r, m, label=r'$m(r)$', color = 'k', linewidth=1.5, linestyle='--')
            if s_type == 3:
                ma = ax2.plot(r, m_A, label=rf'$m_{{{eos_c}}}$', color = colors[c], linewidth=1.5, linestyle='-.')
                mb = ax2.plot(r, m_B, label=r'$m_{{DM}}$', color = colors[3], linewidth=1.5, linestyle='-.')
            ax2.set_ylabel(r'$m$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center', color='k')
            ax2.tick_params(axis='y', colors='k')
                    
            # Add axis lines
            ax1.axhline(0, color='k', linewidth=1.0, linestyle='--')
            ax1.axvline(0, color='k', linewidth=1.0, linestyle='--')
            
            # Set limits
            if False == True:
                ax1.set_xlim(0, 6.41)
                ax1.set_ylim(0, 1e-3)
                ax2.set_ylim(0, 1)
            
            # Configure ticks
            ax1.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True)
            ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True)
            ax1.minorticks_on()
            ax2.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
            ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
            ax2.minorticks_on()
            
            # Configure ticks spacing
            if False == True:
                ax1.set_xticks(np.arange(0, 11.83, 1))
                #ax1.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
                ax1.set_yticks(np.arange(0, 3.51e-5, 0.5e-5))
                #ax1.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)
                ax2.set_yticks(np.arange(0, 1.51, 0.2))
                #ax2.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
            
            # Set thicker axes
            for ax in [ax1, ax2]:
                ax.spines['top'].set_linewidth(1.5)
                ax.spines['right'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['top'].set_color('k')
                ax.spines['right'].set_color('k')
                ax.spines['bottom'].set_color('k')
                ax.spines['left'].set_color('k')
                
            # Add a legend
            #ax1.legend(fontsize=15, frameon=True, fancybox=False, loc = "center left", bbox_to_anchor=(0.01, 0.5), ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
            #ax2.legend(fontsize=15, frameon=True, fancybox=False, loc = "center right", bbox_to_anchor=(0.99, 0.5), ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
                
            # Save the plot as a PDF
            
            if s_type == 1:
                plt.title(rf'TOV solution NS: $EoS={eos_c},$ ${p1_c} = {p1_v}.$', loc='left', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{eos_c}_{p1_c}_{p1_v}.pdf", format="pdf", bbox_inches="tight")
            elif s_type == 2:
                plt.title(rf'TOV solution DMS: 'r'$m_{\chi}$'rf'$={dm_m},$ ${p1_c} = {p1_v}.$', loc='left', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{dm_m}_{p1_c}_{p1_v}.pdf", format="pdf", bbox_inches="tight")
            else:
                DM_ps = {'a':r'\alpha', 'l':r'\lambda'}
                DM_p = DM_ps[p2_c]
                plt.title(rf'TOV solution DANS: $EoS={eos_c},$ 'r'$m_{\chi}$'rf'$={dm_m},$ ${p1_c} = {p1_v},$ ${DM_p} = {p2_v}$', loc='left', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{eos_c}_{dm_m}_{p1_c}_{p1_v}_{p2_c}_{p2_v}.pdf", format="pdf", bbox_inches="tight")
            plt.show()
        
        elif d_type == 1:
            if s_type == 1:
                df = pd.read_csv(f"data\{s_type}_{d_type}_{eos_c}.csv")
            elif s_type == 2:
                df = pd.read_csv(f"data\{s_type}_{d_type}_{dm_m}.csv")
            else:
                df = pd.read_csv(f"data\{s_type}_{d_type}_{eos_c}_{dm_m}_{p2_c}_{p2_v}.csv")
                M_A = df["M_A"]
                M_B = df["M_B"]
            
            R = df["R"]
            M = df["M"]
            
            # Configure the plot
            plt.figure(figsize=(9.71, 6))
            colors = sns.color_palette("Set1", 10)
            eos_colors = {"soft": 0, "middle": 1, "stiff": 2}
            c = eos_colors[eos_c]
            
            # Plot the data
            plt.plot(R, M, label = r'$M(R)$', color = 'k', linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
            if s_type == 3:
                plt.plot(R, M_A, label = rf'$M_{{{eos_c}}}(R)$', color = colors[c], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
                plt.plot(R, M_B, label = r'$M_{{DM}}(R)$', color = colors[3], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
            
            # Add labels and title
            plt.xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
            plt.ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center')
            
            # Set limits
            #plt.xlim(8, 17)
            #plt.ylim(0, 3.5)
            
            # Configure ticks for all four sides
            plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
            plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
            plt.minorticks_on()
            
            # Customize tick spacing
            #plt.gca().set_xticks(np.arange(8, 17.1, 1))  # Major x ticks 
            #plt.gca().set_yticks(np.arange(0, 3.51, 0.5))  # Major y ticks 
            
            # Set thicker axes
            plt.gca().spines['top'].set_linewidth(1.5)
            plt.gca().spines['right'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            plt.gca().spines['left'].set_linewidth(1.5)
            
            # Add a legend
            plt.legend(fontsize=15, frameon=True, fancybox=False, ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
            
            # Save plot as PDF
            if s_type == 1:
                plt.title(rf'MR curve NS: $EoS={eos_c}.$', loc='left', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{eos_c}.pdf", format="pdf", bbox_inches="tight")
            elif s_type == 2:
                plt.title(rf'MR curve DMS: 'r'$m_{\chi}=$'rf'${dm_m}$  $\left[ GeV \right].$', loc='left', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{dm_m}.pdf", format="pdf", bbox_inches="tight")
            else:
                DM_ps = {'a':r'\alpha', 'l':r'\lambda'}
                DM_p = DM_ps[p2_c]
                plt.title(rf'MR curve DANS: $EoS={eos_c},$ 'r'$m_{\chi}$'rf'$={dm_m}$ $\left[ GeV \right],$ ${DM_p} = {p2_v}.$', loc='left', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{eos_c}_{dm_m}_{p2_c}_{p2_v}.pdf", format="pdf", bbox_inches="tight")
                
            plt.show()

        elif d_type == 2:
            # Read data
            df = pd.read_csv(f"data\{s_type}_{d_type}_{eos_c}_{dm_m}_{p1_c}_{p1_v}.csv")
            r = df['r']
            m = df['m']
            p_A = df['p_A']
            p_B = df['p_B']
            m_A = df['m_A']
            m_B = df['m_B']
            
            # Configure the plot
            fig, ax1 = plt.subplots(figsize=(9.71, 6))
            colors = sns.color_palette("Set1", 10)
            eos_colors = {"soft": 0, "middle": 1, "stiff": 2, "DM": 3}
            c = eos_colors[eos_c]
            
            # Plot pressures
            pa = ax1.plot(r, p_A, label=rf'$p_{{{eos_c}}}$', color = colors[c], linewidth=1.5, linestyle='-')
            pb = ax1.plot(r, p_B, label=r'$p_{{DM}}$', color = colors[3], linewidth=1.5, linestyle='-')
            ax1.set_xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
            ax1.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
            ax1.tick_params(axis='y', colors='k')
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
            
            # Plot Mass
            ax2 = ax1.twinx()
            mt = ax2.plot(r, m, label=r'$m(r)$', color = 'k', linewidth=1.5, linestyle='--')
            ma = ax2.plot(r, m_A, label=rf'$m_{{{eos_c}}}$', color = colors[c], linewidth=1.5, linestyle='-.')
            mb = ax2.plot(r, m_B, label=r'$m_{{DM}}$', color = colors[3], linewidth=1.5, linestyle='-.')
            ax2.set_ylabel(r'$m$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center', color='k')
            ax2.tick_params(axis='y', colors='k')
            
            # Add axis lines
            ax1.axhline(0, color='k', linewidth=1.0, linestyle='--')
            ax1.axvline(0, color='k', linewidth=1.0, linestyle='--')
            
            # Set limits
            if False == True:
                ax1.set_xlim(0, 6.41)
                ax1.set_ylim(0, 1e-3)
                ax2.set_ylim(0, 1)
            
            # Configure ticks
            ax1.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True)
            ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True)
            ax1.minorticks_on()
            ax2.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
            ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
            ax2.minorticks_on()
            
            # Configure ticks spacing
            if False == True:
                ax1.set_xticks(np.arange(0, 11.83, 1))
                #ax1.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
                ax1.set_yticks(np.arange(0, 3.51e-5, 0.5e-5))
                #ax1.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)
                ax2.set_yticks(np.arange(0, 1.51, 0.2))
                #ax2.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
            
            # Set thicker axes
            for ax in [ax1, ax2]:
                ax.spines['top'].set_linewidth(1.5)
                ax.spines['right'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['top'].set_color('k')
                ax.spines['right'].set_color('k')
                ax.spines['bottom'].set_color('k')
                ax.spines['left'].set_color('k')
                
            # Add a legend
            #ax1.legend(fontsize=15, frameon=True, fancybox=False, loc = "center left", bbox_to_anchor=(0.01, 0.5), ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
            #ax2.legend(fontsize=15, frameon=True, fancybox=False, loc = "center right", bbox_to_anchor=(0.99, 0.5), ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
                
            # Save the plot as a PDF
            plt.title(
                rf'TOV solution for max $\lambda$ star: $M={p1_v},$ $m_\chi={dm_m}$',
                loc='left',
                fontsize=15,
                fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"preliminary_figures\{s_type}_{d_type}_{eos_c}_{dm_m}_{p1_c}_{p1_v}.pdf", format="pdf", bbox_inches="tight")
            plt.show()
            