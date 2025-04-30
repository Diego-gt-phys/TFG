# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:20:18 2024

Solves the Tolman Oppenheimer Volkoff (TOV) equation for a star of constant density.

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Physical parameters (solar mass = 198847e30 kg & 6.957e8 m)
G = 1.4765679173556 # G in units of solar masses / solar radii
rho = 5e-4  # Solar density in solar units is 0.238732. In IS units it would be 1409.82

def system_of_ODE(r, y):
    """
    Defines the system of differential equations.
    
    Parameters:
        r (float): Independent variable.
        y (list): Dependent variables [m, p].

    Returns:
        list: Derivatives [dm/dr, dp/dr].
    """
    m, p = y
    dm_dr = 4 * np.pi * r**2 * rho
    nominator = - (rho + p) * (G * m + 4 * np.pi * G * r**3 * p)
    denominator = r * (r - 2 * G * m)
    dp_dr = nominator/denominator
    return [dm_dr, dp_dr]

def runge_kutta_4th_order_with_stop(system, y0, r_range, h):
    """
    Solves a system of ODEs using the 4th-order Runge-Kutta method, stopping
    when a condition is met (e.g., p(r) < 0).

    Parameters:
        system (function): The system of ODEs as a function of r and y.
        y0 (list): Initial conditions for the system.
        r_range (tuple): The range of r as (r_start, r_end).
        h (float): Step size.

    Returns:
        tuple: (r_values, y_values) where:
            r_values is an array of r points,
            y_values is an array of solution points for y.
    """
    r_start, r_end = r_range
    r_values = [r_start]
    y_values = [y0]

    r = r_start
    y = np.array(y0)

    while r <= r_end:
        k1 = h * np.array(system(r, y))
        k2 = h * np.array(system(r + h / 2, y + k1 / 2))
        k3 = h * np.array(system(r + h / 2, y + k2 / 2))
        k4 = h * np.array(system(r + h, y + k3))

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Check if the stopping condition is met (p < 0)
        if y_next[1] < y0[1]*1e-10:  # p corresponds to y[1]
            break

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next

    return np.array(r_values), np.array(y_values)

def TOV_solver (y0, r_range, h):
    """
    Solves the TOV equation for a guiven set of initial conditions (m_0, p_0) finding how m and p vary along r.
    
    Parameters:
        y0 (list): Initial conditions for the system [m_0,p_0].
        r_range (tuple): The range of r as (r_start, r_end).
        h (float): Step size.
        
    Returns:
        tuple: (r_values, y_values) where:
            r_values(array): containing the r points
            m_values(array): containing the m points
            p_values(array): containing the p points
    """
    
    # Solve the system with the stopping condition
    r_values, y_values = runge_kutta_4th_order_with_stop(system_of_ODE, y0, r_range, h)

    # Extract solutions
    m_values = y_values[:, 0]
    p_values = y_values[:, 1]
    
    return r_values, m_values, p_values

def M_R_curve (pc_range, r_range, h, n):
    """
    Creates the mass radius curve of a family of stars with the same density by solving the TOV.
    
    Parameters:
        pc_range(tuple): The range of central preassures (pc_start, pc_end)
        r_range (tuple): The range of r as (r_start, r_end)
        h (float): Step size.
        n (int): Number of preassure iterations.
        
    Return:
        R_values (array): Values of R
        M_values (array): Values of M
    """
    pc_start, pc_end = pc_range
    
    R_values = []
    M_values = []
    pc_list = np.logspace(np.log10(pc_start), np.log10(pc_end), n) #DEBUG use logspace for equidistant points
    
    for pc in pc_list:
        r_i, m_i, p_0 = TOV_solver([0, pc], r_range, h)
        
        R_i = r_i[-1]
        M_i = m_i[-1]
        
        R_values.append(R_i)
        M_values.append(M_i)
    
    return np.array(R_values), np.array(M_values)

# Using the functions
r,m,p = TOV_solver((0, 5e-3), (1e-6, 20), 0.001)
print("The star has mass:", m[-1], "and radius:", r[-1])

# Create the plot
plt.figure(figsize=(9.71, 6)) # The image follows the golden ratio
colors = sns.color_palette("Set1", 5) # Generate a color palette
plt.plot(r, p, label = r'A', color = colors[1], linewidth = 2, linestyle = '-', marker = "",  mfc='k', mec = 'k', ms = 6)

# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
#plt.title(r'Soluciones para diferentes $p_c$. $\rho_0 = 10^3$ $[M_{\odot} / R^3_{\odot}]$', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'r $[km]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'p $[M_{\odot}/km^3]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
#plt.xlim(0,7.2)
#plt.ylim(10e-2,2e6)

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
#plt.gca().set_xticks(np.arange(0, 7.21, 0.5))  # Major x ticks 
#plt.gca().set_yticks(np.arange(0, 1.6e6, 0.2e6))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=15, frameon=False) #  loc='upper right',

# Save the plot as a PDF
#plt.savefig("Constant_eos_MR.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()
