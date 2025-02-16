# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:20:18 2024

Solves the Tolman Oppenheimer Volkoff (TOV) equation for a star whose equation of state (eos) is in guiven by a dataset.

@author: Diego García Tejada
"""

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
def create_eos(Type, p_range, n, plot=False):
    """
    Function that generates an excel containing the data of the equation of state

    Parameters
    ----------
    Type : string
        Type of eos that you require. For a constant density star use 'constant', for a polytropic eos use 'polytropic'.
    p_range : touple
        Vector that contains the initial and final preassures. The structure is: (pi, pf).
    n : interger
        Number of data points you want the data to have.
    plot : Boolian, optional
        Wether or not the code should make a plot of the created eos. The default is False.

    Returns
    -------
    None.

    """
    
    pi, pf = p_range
    
    if Type == "constant":
        p = np.linspace(pi, pf, n)
        rho = np.zeros(n) +  2.954457325e-4
            
    elif Type == "polytropic":
        K=150
        gamma=2
        p = np.linspace(pi, pf, n)
        rho = (p / K) ** (1 / gamma)
    
    data = pd.DataFrame({'Density': rho, 'Pressure': p})
    data.to_excel("data.xlsx", index=False)
    
    if plot==True:
        plt.figure(figsize=(8, 8))
        plt.plot(p, rho, label = r'$p(\rho)$', color = "black", linewidth = 2, linestyle = '-', marker = '.', markersize = 10)
        #plt.title(r'Created eos', loc='center', fontsize=20, fontweight='bold')
        plt.xlabel(r'$p$ $\left[ M_{\odot}/km^3 \right]$', fontsize=15, loc='center', fontweight='bold')
        plt.ylabel(r'$\rho$ $\left[ M_{\odot}/km^3 \right]$', fontsize=15, loc='center', fontweight='bold')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=12, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
        plt.minorticks_on()
        plt.gca().spines['top'].set_linewidth(1.7)
        plt.gca().spines['right'].set_linewidth(1.7)
        plt.gca().spines['bottom'].set_linewidth(1.7)
        plt.gca().spines['left'].set_linewidth(1.7)
        plt.legend(fontsize=17, frameon=False)
        plt.show()
        
    return None

def eos(p):
    """
    Guiven the arrays p_data and rho_data which contain the information for the equation of state, this funtion interpolates the value of rho for a guiven  p. 

    Parameters
    ----------
    p : float
        Preassure at which we want to evaluate the eos.

    Returns
    -------
    rho : float
        Density associated to the preassure guiven.

    """
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p)
    
    return rho

def system_of_ODE(r, y):
    """
    This functions calculate the numerical values for dm/dr, dp/dr for the differential equations that deffine stellar structure.

    Parameters
    ----------
    r : float
        Radius were we are evaluating the derivatives.
    y : tuple
        Vector containig the information of mass and pressure at a point r. The structure is: (m, p).

    Returns
    -------
    list
        Vector containing the information of the derivative values.

    """
    m, p = y
    rho = eos(p)
    dm_dr = 4 * np.pi * r**2 * rho
    nominator = - (rho + p) * (G * m + 4 * np.pi * G * r**3 * p)
    denominator = r * (r - 2 * G * m)
    dp_dr = nominator/denominator
    
    return [dm_dr, dp_dr]

def runge_kutta_4th_order_with_stop(system, y0, r_range, h):
    """
    Solves a system of ODEs using the 4th-order Runge-Kutta method, stopping
    when the condition of preassure is met.

    Parameters
    ----------
    system : function
        System of ODEs as a function of r and y.
    y0 : list
        Initial conditions of the system. The structure is: [mc, pc].
    r_range : tuple
        The range of r as (r_start, r_end).
    h : float
        Step size of integration.

    Returns
    -------
    tuple
        (r_values, y_values) where:
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
        if y_next[1] < y0[1]*1e-10:  # Beware of tail
            break

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next

    return (np.array(r_values), np.array(y_values))

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

###############################################################################
# Calculate the data
###############################################################################
# Read the data
data = pd.read_excel("soft.xlsx")
rho_data = data['Density'].values
p_data = data['Pressure'].values

# Find the solution for the TOV equation.
r,m,p = TOV_solver([0,1.6191e-5], (1e-6,20), 0.001)
###############################################################################
# Plot the data
###############################################################################
plt.figure(figsize=(9.71, 6)) # The image follows the golden ratio
colors = sns.color_palette("Set1", 5) # Generate a color palette
plt.plot(r, p, label = r'soft', color = colors[0], linewidth = 2, linestyle = '-', marker = '', mfc='k', mec = 'k', ms = 6) # marker = '', mfc='k', mec = 'k', ms = 6

# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
plt.title(r'Solución de TOV para la EOS: soft', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'r $\left[km\right]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'p $\left[M_{\odot}/km^3\right]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
#plt.xlim(8, 17)
#plt.ylim(0, 3.3)

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
#plt.gca().set_xticks(np.arange(9, 15.1, 0.5))  # Major x ticks 
#plt.gca().set_yticks(np.arange(0, 3.51, 0.5))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=15, frameon=False) #  loc='upper right',

# Save the plot as a PDF
#plt.savefig("TOV.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()