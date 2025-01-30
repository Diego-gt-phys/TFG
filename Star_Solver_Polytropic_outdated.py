# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:20:18 2024

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Physical parameters (solar mass = 198847e30 kg & 6.957e8 m)
G = 2.12242e-6 # G in units of solar masses / solar radii
#rho = 1e3  # Solar density in solar units is 0.238732. In IS units it would be 1409.82

def eos_example(p):
    """
    Example equation of state: Polytropic EOS.
    
    Parameters:
        p (float): Pressure.

    Returns:
        float: Corresponding density rho(p).
    """
    K = 1  # Polytropic constant
    n = 1.5   # Polytropic index
    #rho = (p / K) ** (1 / (n + 1))
    rho = 1e3 + 1*(p / K) ** (1 /n)
    return rho

def system_of_ODE(r, y, eos):
    """
    Defines the system of differential equations for the TOV equation.
    
    Parameters:
        r (float): Independent variable.
        y (list): Dependent variables [m, p].
        eos (function): Equation of state function rho(p).

    Returns:
        list: Derivatives [dm/dr, dp/dr].
    """
    m, p = y
    rho = eos_example(p)  # Compute density from pressure using the EOS
    dm_dr = 4 * np.pi * r**2 * rho
    nominator = - (rho + p) * (G * m + 4 * np.pi * G * r**3 * p)
    denominator = r * (r - 2 * G * m)
    dp_dr = nominator / denominator
    return [dm_dr, dp_dr]

def runge_kutta_4th_order_with_stop(system, y0, r_range, h, eos):
    # Modify to pass the EOS to the system function
    r_start, r_end = r_range
    r_values = [r_start]
    y_values = [y0]

    r = r_start
    y = np.array(y0)

    while r <= r_end:
        k1 = h * np.array(system(r, y, eos))
        k2 = h * np.array(system(r + h / 2, y + k1 / 2, eos))
        k3 = h * np.array(system(r + h / 2, y + k2 / 2, eos))
        k4 = h * np.array(system(r + h, y + k3, eos))

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Check if the stopping condition is met (p < 0)
        if y_next[1] < 0:  # p corresponds to y[1]
            break

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next

    return np.array(r_values), np.array(y_values)

def TOV_solver(y0, r_range, h, eos):
    # Modify to pass the EOS to the ODE solver
    r_values, y_values = runge_kutta_4th_order_with_stop(system_of_ODE, y0, r_range, h, eos)
    m_values = y_values[:, 0]
    p_values = y_values[:, 1]
    return r_values, m_values, p_values

def M_R_curve(pc_range, r_range, h, n, eos):
    pc_start, pc_end = pc_range
    R_values = []
    M_values = []
    pc_list = np.logspace(np.log10(pc_start), np.log10(pc_end), n)

    for pc in pc_list:
        r_i, m_i, p_0 = TOV_solver([0, pc], r_range, h, eos)
        R_i = r_i[-1]
        M_i = m_i[-1]
        R_values.append(R_i)
        M_values.append(M_i)
    
    return np.array(R_values), np.array(M_values)

# Using the functions
R, m, M = TOV_solver([0, 1e7], (1e-3,10), 0.005, eos_example)
#R, M = M_R_curve((1e-4,1e7), (1e-3,10), 0.005, 200)

# Create the plot
plt.figure(figsize=(9.71, 6)) # The image follows the golden ratio
colors = sns.color_palette("Set1", 5) # Generate a color palette
plt.plot(R, M, label = r'Curva 1', color = colors[0], linewidth = 2, linestyle = '-', marker = "", markersize=5)

# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
#plt.title(r'Soluciones para diferentes $p_c$. $\rho_0 = 10^3$ $[M_{\odot} / R^3_{\odot}]$', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'R $[R_{\odot}]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'M $[M_{\odot}]$', fontsize=15, loc='center', fontweight='bold')
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
plt.savefig("Fig.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()
