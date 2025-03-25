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
DM_mass = 1 # Mass of dark matter particle in GeV
Gamma = 5/3 # Polytropic coeficient for a degenetare (T=0) IFG
K = ((DM_mass)**(-8/3))*8.0165485819726 # Polytropic constant for a degenetare (T=0) IFG

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

def get_inputs(mode_DB, data_type_DB, eos_choice_DB, param_choice_DB, param_value_DB, central_pressure_DB):
    """
    Prompts the user for input parameters to configure the DANTE solver.

    Parameters
    ----------
    Same deffinition as the returns.
    This parameters are used as configuration data for the DEBUG mode.

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
    eos_choice : str or None
        'soft', 'mid', or 'stiff' for the equation of state.
    param_choice : str or None
        'a' for Alpha, 'l' for Lambda, or None if not applicable.
    param_value : float or None
        The value of the chosen DM parameter or None if not applicable.
    central_pressure : float or None
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
                    eos_options = {1: 'soft', 2: 'middle', 3: 'stiff'}
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
        mode, data_type, eos_choice, param_choice, param_value, central_pressure = (mode_DB, data_type_DB, eos_choice_DB, param_choice_DB, param_value_DB, central_pressure_DB)
        
    return mode, data_type, eos_choice, param_choice, param_value, central_pressure

def save_TOV_data_1f (d_type, eos_c, p_c):
    """
    Solves the TOV equation for 1 fluid and saves the data as a .csv file.
    It has the choice to solve both a NS of a DM star.
    It saves the data in a folder called data. This folder must be in the same repository as this code.

    Parameters
    ----------
    d_type : int
        if d_type == 1, then we are solving a neutron star.
        if d_type == 2, then we are solving a dark matter star.
    eos_c : str
        Equation of State of the neutron star. The options are 'soft', 'middle', 'stiff'.
        Due to how the code is structure, for a dark matter star, this EoS must be specified but it has no impact.
    p_c : float
        Pressure at r=0 of the fluid.

    Returns
    -------
    df : Pandas DataFrame
        DF containing all the data of the solved model.
    """
    data = {}
    
    if d_type == 1: # Solution to the TOV for fluid A
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0,p_c,0,0,0), (1e-6, 100), 1e-3)
    
    elif d_type == 2: # Solution to the TOV for fluid B
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0,0,p_c,0,0), (1e-6, 100), 1e-3)
    
    data["r"] = r
    data["m"] = m
    data["p_A"] = p_A
    data["p_B"] = p_B
    data["m_A"] = m_A
    data["m_B"] = m_B
    data["R_A"] = R_A
    df = pd.DataFrame(data)
    
    if d_type == 1:
        df.to_csv(f"data\{d_type}_{eos_c}_{p_c}.csv", index=False)
        
    elif d_type == 2:
        df.to_csv(f"data\{d_type}_{p_c}_{DM_mass}.csv", index=False)
    
    return df
    
def save_TOV_data_2f (eos_c, param_c, param_val, p_c):
    """
    Solves the TOV equation for 2 fluid star and saves the data as a .csv file.
    The amount of DM is controlled by the parameter choice: alpha or lambda.
        
    Parameters
    ----------
    eos_c : str
        Equation of State of the neutron star. The options are 'soft', 'middle', 'stiff'.
    param_c : str
        Choice of paremeter. The options are alpha(a) or lambda(l).
    param_val : float
        Value of previuslychosen paremeter.
    p_c : float
        Value of central pressure of fluid A.
    
    Returns
    -------
    df : Pandas DataFrame
        DF containing all the data of the solved model.
    """
    data = {}
    if param_c == "a": # Usamos parámetro alpha
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0,p_c, param_val*p_c,0,0), (1e-6, 100), 1e-3)
        
    elif param_c == "l":
        # Find the alpha_target
        alpha = find_lambda(p_c, param_val)
        print(f"\nValue of alpha found: {alpha}")
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0,p_c, alpha*p_c,0,0), (1e-6, 100), 1e-3)
    
    data["r"] = r
    data["m"] = m
    data["p_A"] = p_A
    data["p_B"] = p_B
    data["m_A"] = m_A
    data["m_B"] = m_B
    data["R_A"] = R_A
    df = pd.DataFrame(data)
        
    df.to_csv(f"data\{d_type}_{eos_c}_{param_c}_{param_val}_{p_c}_{DM_mass}.csv", index=False)
    
    return df

def save_MR_data (eos_c, param_c, param_val):
    """
    Calculates the MR curve for a 2 fluid star and saves the data inside a .csv file.
    The amount of fluid B is controled by the parameter choice.
    
    Parameters
    ----------
    eos_c : str
        Equation of State of the neutron star. The options are 'soft', 'middle', 'stiff'.
    param_c : str
        Choice of paremeter. The options are alpha(a) or lambda(l).
    param_val : float
        Value of previuslychosen paremeter.
    
    Returns
    -------
    df : Pandas DataFrame
        DF containing all the data of the solved model.
    """
    data = {}
    pc_range = PCS[eos_c]
    
    if param_c == 'a':
        R, M, M_A, M_B = MR_curve(pc_range, param_val, (1e-6, 100), 1e-3, 20)
    elif param_c == 'l':
        R, M, M_A, M_B = MR_curve_lambda(pc_range, param_val, (1e-6, 100), 1e-3, 20)
        
    data["R"] = R
    data["M"] = M
    data["M_A"] = M_A
    data["M_B"] = M_B
    df = pd.DataFrame(data)
    df.to_csv(f"data\{d_type}_{eos_c}_{param_c}_{param_val}_{DM_mass}.csv", index=False)
    
    return df

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
    
###############################################################################
# Define the parameters
###############################################################################

print("Welcome to DANTE: the Dark-matter Admixed Neutron-sTar solvEr.")

mode, d_type, eos_c, param_c, param_val, p_c = get_inputs(1, 3, 'soft', 'a', 0.11970180169768445, 7.905105704757548e-05)

print(f"\nUser Inputs: {mode}, {d_type}, '{eos_c}', '{param_c}', {param_val}, {p_c}")
print("")

###############################################################################
# Create the data
###############################################################################

if mode == 0:
    
    if d_type == 2:
        eos_c = 'soft'
    
    eos_data = pd.read_excel(f"data_eos\eos_{eos_c}.xlsx")
    rho_data = eos_data['Density'].values
    p_data = eos_data['Pressure'].values
    
    if d_type in [1,2]:
        df = save_TOV_data_1f(d_type, eos_c, p_c)
    elif d_type == 3:
        df = save_TOV_data_2f(eos_c, param_c, param_val, p_c)
    elif d_type == 4:
        df = save_MR_data(eos_c, param_c, param_val)
    
    print(df)
    print("Data Saved.")               

###############################################################################
# Plot the data
###############################################################################

if mode == 1:
    
    if d_type in [1,2]: # Solution to the TOV for 1 fluid
            
        #Read the data
        if d_type == 1:
            df = pd.read_csv(f"data\{d_type}_{eos_c}_{p_c}.csv")
            r = df["r"]
            m = df["m"]
            p = df["p_A"]
            
        elif d_type == 2:
            df= pd.read_csv(f"data\{d_type}_{p_c}_{DM_mass}.csv")
            r = df["r"]
            m = df["m"]
            p = df["p_B"]
            eos_c = "DM"
            
        # Scale factors
        p_scale = 5
        m_scale = 1
        
        # Configure the plot
        plt.figure(figsize=(9.71, 6))
        colors = sns.color_palette("Set1", 10)
        eos_colors = {"soft": 0, "middle": 1, "stiff": 2, "DM": 3}
        c = eos_colors[eos_c]
        
        # Plot the data    
        plt.plot(r, p*10**p_scale, label = rf'$p_{{{eos_c}}}(r)$', color = colors[c], linewidth = 1.5, linestyle = '-') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m*m_scale, label = rf'$m_{{{eos_c}}}(r)$', color = colors[c], linewidth = 1.5, linestyle = '-.') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        
        # Set the axis to logarithmic scale
        #plt.xscale('log')
        #plt.yscale('log')
        
        # Add labels and title
        if d_type == 1:
            plt.title(rf'TOV solution for the {eos_c} eos.', loc='left', fontsize=15, fontweight='bold')
        elif d_type == 2:
            plt.title(rf'TOV solution for a {eos_c} star with: $m_{{\chi}}={DM_mass} [GeV]$', loc='left', fontsize=15, fontweight='bold')
        plt.xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
        plt.ylabel(rf'$p\cdot 10^{p_scale}$' r'$\left[ M_{{\odot}}/km^3\right]$ & $m$ $\left[ M_{\odot}\right]$', fontsize=15, loc='center')
        plt.axhline(0, color='k', linewidth=1.0, linestyle='--')  # x-axis
        plt.axvline(0, color='k', linewidth=1.0, linestyle='--')  # y-axis
        
        # Set limits
        #plt.xlim(0, 23.15)
        #plt.ylim(0, 1)
        
        # Add grid
        #plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Configure ticks for all four sides
        plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
        plt.minorticks_on()
        
        # Customize tick spacing for more frequent ticks on x-axis
        #plt.gca().set_xticks(np.arange(0, 23.15, 2))  # Major x ticks 
        #plt.gca().set_yticks(np.arange(0, 1, 0.1))  # Major y ticks 
        
        # Set thicker axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        
        # Add a legend
        plt.legend(fontsize=12, frameon=False, ncol = 1) #  loc='upper right',
        
        # Save the plot as a PDF
        if d_type == 1:
            plt.savefig(f"preliminary_figures\{d_type}_{eos_c}_{p_c}.pdf", format="pdf", bbox_inches="tight")
            
        elif d_type == 2:
            plt.savefig(f"preliminary_figures\{d_type}_{p_c}_{DM_mass}.pdf", format="pdf", bbox_inches="tight")
        
        plt.tight_layout()
        plt.show()
    
    if d_type == 3: # Solution to the TOV for 2 fluid
        
        # Read the data
        df = pd.read_csv(f"data\{d_type}_{eos_c}_{param_c}_{param_val}_{p_c}_{DM_mass}.csv")
        
        r = df["r"]
        p_A = df["p_A"]
        p_B = df["p_B"]
        m = df["m"]
        m_A = df["m_A"]
        m_B = df["m_B"]
        
        # Scale factors
        p_scale = 4
        m_scale = 1
        
        # Configure the plot
        plt.figure(figsize=(9.71, 6))
        colors = sns.color_palette("Set1", 10)
        eos_colors = {"soft": 0, "middle": 1, "stiff": 2}
        c = eos_colors[eos_c]
        
        # Plot the data
        plt.plot(r, p_A*10**p_scale, label = rf'$p_{{{eos_c}}}(r)$', color = colors[c], linewidth = 1.5, linestyle = '-') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m_A*m_scale, label = rf'$m_{{{eos_c}}}(r)$', color = colors[c], linewidth = 1.5, linestyle = '-.') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, p_B*10**p_scale, label = r'$p_{DM}(r)$', color = colors[3], linewidth = 1.5, linestyle = '-') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m_B*m_scale, label = r'$m_{DM}(r)$', color = colors[3], linewidth = 1.5, linestyle = '-.') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m*m_scale, label = r'$m(r)$', color = 'k', linewidth = 1.5, linestyle = '--') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        
        # Set the axis to logarithmic scale
        #plt.xscale('log')
        #plt.yscale('log')
        
        # Add labels and title
        if param_c == 'a':
            param_val_round = round(param_val, 4)
            plt.title(rf'TOV solution for a DANS with: $EoS={eos_c},$ $\alpha = {param_val_round},$ $m_{{\chi}}={DM_mass}[GeV]$', loc='left', fontsize=15, fontweight='bold')
        elif param_c == 'l':
            plt.title(rf'TOV solution for a DANS with: $EoS={eos_c},$ $\lambda = {param_val},$ $m_{{\chi}}={DM_mass}[GeV]$', loc='left', fontsize=15, fontweight='bold')
        plt.xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
        if m_scale == 1:
            plt.ylabel(rf'$p\cdot 10^{p_scale}$' r'$\left[ M_{{\odot}}/km^3\right]$ & $m$ $\left[ M_{\odot}\right]$', fontsize=15, loc='center')
        else:
            plt.ylabel(rf'$p\cdot 10^{p_scale}$' r'$\left[ M_{{\odot}}/km^3\right]$ & ' rf'$m \cdot {m_scale}$' r'$\left[ M_{\odot}\right]$', fontsize=15, loc='center')
        plt.axhline(0, color='k', linewidth=1.0, linestyle='--')  # x-axis
        plt.axvline(0, color='k', linewidth=1.0, linestyle='--')  # y-axis
        
        # Set limits
        plt.xlim(0, 9.6)
        plt.ylim(0, 1)
        
        # Add grid
        #plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Configure ticks for all four sides
        plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
        plt.minorticks_on()
        
        # Customize tick spacing for more frequent ticks on x-axis
        plt.gca().set_xticks(np.arange(0.5, 9.6, 0.5))  # Major x ticks 
        plt.gca().set_yticks(np.arange(0, 1.01, 0.1))  # Major y ticks 
        
        # Set thicker axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        
        # Add a legend
        plt.legend(fontsize=12, frameon=False, ncol = 3, loc='upper center') #  loc='upper right',
        
        # Save the plot as a PDF
        plt.savefig(f"preliminary_figures\{d_type}_{eos_c}_{param_c}_{param_val}_{p_c}_{DM_mass}.pdf", format="pdf", bbox_inches="tight")
        
        plt.tight_layout()
        plt.show()
        
    if d_type == 4: # MR curves
        # Read data
        df = pd.read_csv(f"data\{d_type}_{eos_c}_{param_c}_{param_val}_{DM_mass}.csv")
        R = df["R"]
        M = df["M"]
        M_A = df["M_A"]
        M_B = df["M_B"]
        
        # Configure the plot
        plt.figure(figsize=(9.71, 6))
        colors = sns.color_palette("Set1", 10)
        eos_colors = {"soft": 0, "middle": 1, "stiff": 2}
        c = eos_colors[eos_c]
        
        # Plot the data
        plt.plot(R, M, label = r'$M(R)$', color = 'k', linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
        plt.plot(R, M_A, label = rf'$M_{{{eos_c}}}(R)$', color = colors[c], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
        plt.plot(R, M_B, label = r'$M_{{DM}}(R)$', color = colors[3], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
        
        # Add labels and title
        if param_c == 'a':
            plt.title(rf'MR curve for a DANS with: $EoS={eos_c},$ $\alpha = {param_val},$ $m_{{\chi}}=1[GeV]$', loc='left', fontsize=15, fontweight='bold')
        elif param_c == 'l':
            plt.title(rf'MR curve for a DANS with: $EoS={eos_c},$ $\lambda = {param_val},$ $m_{{\chi}}=1[GeV]$', loc='left', fontsize=15, fontweight='bold')
        plt.xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
        plt.ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center')
        
        # Set limits
        plt.xlim(8, 17)
        plt.ylim(0, 3.5)
        
        # Add grid
        #plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Configure ticks for all four sides
        plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
        plt.minorticks_on()
        
        # Customize tick spacing for more frequent ticks on x-axis
        plt.gca().set_xticks(np.arange(8, 17.1, 1))  # Major x ticks 
        plt.gca().set_yticks(np.arange(0, 3.51, 0.5))  # Major y ticks 
        
        # Set thicker axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        
        # Add a legend
        plt.legend(fontsize=12, frameon=False, ncol = 1) #  loc='upper right',
        
        # Save the plot as a PDF
        plt.savefig(f"preliminary_figures\{d_type}_{eos_c}_{param_c}_{param_val}_{DM_mass}.pdf", format="pdf", bbox_inches="tight")
        
        plt.tight_layout()
        plt.show()
        
        