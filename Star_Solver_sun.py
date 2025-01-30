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
rho = 0.238732  # Solar density in solar units is 0.238732. In IS units it would be 1409.82

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
        if y_next[1] < 0:  # p corresponds to y[1]
            break

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next

    return np.array(r_values), np.array(y_values)

# Initial conditions
y0 = [0, 2.53346e-7]  # Initial conditions for m and p
r_range = (1e-8, 1.1)  # Solve from r=0 to r=5
h = 0.001  # Step size

# Solve the system with the stopping condition
r_values, y_values = runge_kutta_4th_order_with_stop(system_of_ODE, y0, r_range, h)

# Extract solutions
m_values = y_values[:, 0] * 2e-7
p_values = y_values[:, 1]

# Generate a color palette
colors = sns.color_palette("Set1", 5)

# Create the plot
plt.figure(figsize=(9.71, 6)) # The image follows the golden ratio
plt.plot(r_values, p_values, label = r'$p(r)$', color = colors[0], linewidth = 1.5, linestyle = '-')
plt.plot(r_values, m_values, label = r'$m(r) \cdot 0.5 \cdot 10^{7}$', color = colors[1], linewidth = 1.5, linestyle='-.')

# Add labels and title
plt.title(r'Estrella de densidad constante', loc='left', fontsize=14, fontweight='bold')
plt.xlabel(r'r $[R_{\odot}]$', fontsize=12, loc='center', fontweight='bold')
plt.ylabel(r'm(r) $[M_{\odot}]$ $\cdot 0.5 \cdot 10^{7}$ & p(r) $[M_{\odot} / R_{\odot}^3]$', fontsize=12, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
plt.xlim(0,1)
plt.ylim(0,2.6e-7)

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
plt.gca().set_xticks(np.arange(0.1, 1.01, 0.1))  # Major x ticks 
#plt.gca().set_yticks(np.arange(1, 8.1, 1))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=15, frameon=False) #  loc='upper right',

# Save the plot as a PDF
#plt.savefig("Sol_de_densidad_constante.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()