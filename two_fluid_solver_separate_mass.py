# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:34:28 2025

Solves TOV equation for a star formed of two fluids, A and B.

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
        if y_next[1] <= y0[1]*1e-10 and y_next[2] <= y0[2]*1e-10: # If both pressures drop to 0 then the star stops there.
            break
        
        elif y_next[1] <= y0[1]*1e-10:  # If fluid A's pressure drops to 0, keep it that way
            y_next[1] = 0
            
        elif y_next[2] <= y0[2]*1e-10:  # If fluid B's pressure drops to 0, keep it that way
            y_next[2] = 0

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next

    return (np.array(r_values), np.array(y_values))

def TOV_solver(y0, r_range, h):
    """
    Using a 4th Runge Kutta method, it solves the TOV for a 2 perfect fluid star.
    It guives the mass and preassure values for both fluids in all of the stars radius.

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
    m_values : array
        Array containing the different values of m(r).
    p_A_values : array
        Array containing the different values of P_A(r).
    p_B_values : array
        Array containing the different values of P_B(r).

    """
    
    r_values, y_values = RK4O_with_stop(y0, r_range, h)
    
    m_values = y_values[:, 0]
    p_A_values = y_values[:, 1]
    p_B_values = y_values[:, 2]
    m_A_values = y_values[:, 3]
    m_B_values = y_values[:, 4]

    return (r_values, m_values, p_A_values, p_B_values, m_A_values, m_B_values)

###############################################################################
# Simulation
###############################################################################

p_c = 0.95e-4
alpha = 0.2
y0 = (0, p_c, alpha*p_c, 0, 0)

r, m, p_A, p_B, m_A, m_B = TOV_solver(y0, (1e-6, 50), 1e-3)

print("M=", m[-1])
print("R=", r[-1])

###############################################################################
# Plot
###############################################################################

plt.figure(figsize=(9.71, 6))
plt.plot(r, p_A * 1e4, label = r'$p_A(r) \cdot 10^4$', color = 'r', linewidth = 1.5)
plt.plot(r, p_B * 1e4, label = r'$p_B(r) \cdot 10^4$', color = 'b', linewidth = 1.5)
plt.plot(r, m*0.8, label = r'$0.8 \cdot m(r)$', color = 'g', linewidth = 1.5, linestyle = '-')
plt.plot(r, m_A*0.8, label = r'$0.8 \cdot m_A(r)$', color = 'r', linewidth = 1, linestyle = '-.')
plt.plot(r, m_B*0.8, label = r'$0.8 \cdot m_B(r)$', color = 'b', linewidth = 1, linestyle = '-.')

# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
plt.xlabel(r'r $\left[km\right]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'$\mathbf{p}\cdot10^4$ $\left[M_{\odot}/km^3\right]$ ; m $\left[M_{\odot}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
plt.xlim(0, 8)
plt.ylim(0, 1)

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=9, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=5, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
plt.gca().set_xticks(np.arange(0, 8.1, 1))  # Major x ticks 
plt.gca().set_yticks(np.arange(0, 1.1, 0.1))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=15, frameon=False, ncol = 2, loc = 'upper right') #  loc='upper right',

# Save the plot as a PDF
plt.savefig("2_fluid_TOV.pdf", format="pdf", bbox_inches="tight")

plt.show()
