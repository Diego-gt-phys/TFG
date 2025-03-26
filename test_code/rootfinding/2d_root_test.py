# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 08:47:34 2025

find the roots of a vector function

@author: Diego García Tejada
"""

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
# Essental Functions
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

###############################################################################
# TEST
###############################################################################
"""
def f(x):
    p_c, alpha = x
    r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, p_c, alpha*p_c, 0, 0), (1e-6, 100), 1e-3)
    M = m[-1]
    l = m_B[-1]/m[-1]
    
    return M-1, l-0.05

eos = "soft"
eos_data = pd.read_excel(f"eos_{eos}.xlsx")
rho_data = eos_data['Density'].values
p_data = eos_data['Pressure'].values

x0 = [1e-4, 0.1]

res_M, res_l = f(x0)

print("This guess gives the following residuals:")
print(f"M:{res_M}")
print(f"L:{res_l}")

sol = opt.root(f, x0, method='hybr')

print(sol)
"""


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
        
M_target, l_target = (1, 0.2)
eos = "soft"
eos_data = pd.read_excel(f"eos_{eos}.xlsx")
rho_data = eos_data['Density'].values
p_data = eos_data['Pressure'].values

pc, alpha = find_sc(M_target, l_target)

print(f"For a star of {M_target} solar mass, and {l_target} of DM. The initial conditions are:", pc, alpha)
