# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:18:11 2025

Toy version of DANTE: Dark-matter Admixed Neutron-sTar solvEr

Calculates the MR curves for a star of constant density.

author: Diego García Tejada
"""

###############################################################################
# Imports and units
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as opt
import multiprocessing as mp
from tqdm import tqdm

# Physical parameters (solar mass = 1.98847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses

###############################################################################
# Define the functions
###############################################################################
def eos (p, dm_m=0.939565):
    """
    Given a pressure p, gives the value of density rho in acordance to the EoS.

    Parameters
    ----------
    p : float
        Pressure.
    dm_m : float
        Mass of the particle ing GeV.

    Returns
    -------
    TYPE
        Density(p).

    """
    if p <= 0:
        return 0
    
    Gamma = 5/3
    K = ((dm_m)**(-8/3))*8.0164772576254
    rho = (p / K) ** (1/Gamma)
        
    return rho

def system_of_ODE (r, y):
    """
    Function that calculates the derivatives of m and the pressure. This function is used for the runge-Kutta method.

    Parameters
    ----------
    r : float
        radius inside of the star.
    y : tuple
        (m, p), where m is the mass, p is the preassure of the fluid, evaluated at point r. 
    
    Returns
    -------
    dm_dr : float
        rate of change of the mass.
    dp_dr : float
        rate of change of the pressure of fluid.

    """
    m, p = y
    rho = eos(p)
    
    dm_dr = 4 * np.pi * (rho) * r**2
    dphi_dr = (G * m + 4 * np.pi * G * r**3 * (p)) / (r * (r - 2 * G * m))
    dp_dr = -(rho + p) * dphi_dr
    
    return (dm_dr, dp_dr)
    
def RK4O_with_stop (y0, r_range, h):
    """
    Function that integrates the y vector using a Runge-Kutta 4th orther method.
    Due to the physics of our problem. The function is built with a condition that doesn't allow negative pressures. 

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_c)
    r_range : tuple
        Range of integratio: (r_0, r_max)
    h : float
        Step size of integration.
        
    Returns
    -------
    r_values : array
        Array containing the different values of r.
        
    y_values : array
        Array containig the solutions for the vector y: (m_values, p_values).
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
        if y_next[1] <= y0[1]*1e-10: # If both pressures drop to 0 then the star stops there.
            break
        
        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next
        
    return (np.array(r_values), np.array(y_values))

def TOV_solver (y0, r_range, h):
    """
    Using a 4th Runge Kutta method, it solves the TOV for a perfect fluid star.
    It guives the mass and preassure values in all of the stars radius.

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_c).
    r_range : tuple
        Range of integratio: (r_0, r_max).
    h : float
        Step size of integration.

    Returns
    -------
    r_values : array
        Array containing the different values of r.
    m_values : array
        Array containing the different values of m(r).
    p_values : array
        Array containing the different values of p(r).
    """
    
    r_values, y_values = RK4O_with_stop(y0, r_range, h)
    
    m_values = y_values[:, 0]
    p_values = y_values[:, 1]
    
    return (r_values, m_values, p_values)

def find_pc (M_target):
    """
    Finds the value of central pressure that creates a star of mass M_target.

    Parameters
    ----------
    M_target : float
        Desired mass.
    eos_type : int
        Type of EoS to use. 0 for an EoS of constant density. 1 for a Polytropic EoS.

    Raises
    ------
    ValueError
        When the root finding algorithm does not converge.

    Returns
    -------
    float
        value of pc that creates a star of mass M_target.
    """
    def f(pc):
        r, m, p = TOV_solver((0, pc), (1e-6, 100), 1e-3)
        M = m[-1]
        return (M - M_target) / M_target
    
    pc_guess = M_target*1.4e-3/3.3
    result = opt.root_scalar(f, x0=pc_guess, method='secant', x1=pc_guess*1.1, xtol=1e-4)
    if result.converged:
        return result.root
    else:
        raise ValueError("Root-finding did not converge")
      
def solve_single_star_MR_curve(args):
    pc, r_range, h = args
    r_i, m_i, p_0 = TOV_solver([0, pc], r_range, h)
    
    R_i = r_i[-1]
    M_i = m_i[-1]
    
    return R_i, M_i

def MR_curve(pc_range, r_range, h, n):
    """
    Computes the mass-radius (M-R) curve for a compact star by solving the 
    Tolman-Oppenheimer-Volkoff (TOV) equations over a range of central pressures.
    Parallelized using multiprocessing.

    Parameters
    ----------
    pc_range : tuple
        Tuple (pc_start, pc_end) specifying the range of central pressures.
    r_range : tuple
        Range of integratio: (r_0, r_max).
    h : float
        Step size used in the numerical integration.
    n : int
        Number of points to sample between pc_start and pc_end (logarithmically spaced).

    Returns
    -------
    R_values : np.ndarray
        Array of stellar radii corresponding to each central pressure.
    M_values : np.ndarray
        Array of stellar masses corresponding to each central pressure.
    """
    pc_start, pc_end = pc_range
    pc_list = np.geomspace(pc_start, pc_end, n)
    args_list = [(pc, r_range, h) for pc in pc_list]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(solve_single_star_MR_curve, args_list), total=n, desc="Processing Stars"))
        
    R_values, M_values = zip(*results)
    return np.array(R_values), np.array(M_values)

if __name__ == '__main__':

###############################################################################
# Calculate the data
###############################################################################

    r, m, p = TOV_solver((0, 1e-5), (1e-6, 100), 1e-3)

###############################################################################
# Plot the data
###############################################################################
    
    fig, ax1 = plt.subplots(figsize=(9.71, 6))
    colors = sns.color_palette("Set1", 10)
    
    # Plot p
    ax1.plot(r, p, label=r'$p(r)$', color = colors[0], linewidth=1.5, linestyle='-')
    ax1.set_xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
    ax1.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
    ax1.tick_params(axis='y', colors='k')
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
    
    # Plot m
    ax2 = ax1.twinx()
    ax2.plot(r, m, label=r'$m(r)$', color = colors[0], linewidth=1.5, linestyle='--')
    ax2.set_ylabel(r'$m$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center', color='k')
    ax2.tick_params(axis='y', colors='k')
    
    # Add axis lines
    ax1.axhline(0, color='k', linewidth=1.0, linestyle='--')
    ax1.axvline(0, color='k', linewidth=1.0, linestyle='--')
    
    # Configure ticks
    ax1.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True)
    ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True)
    ax1.minorticks_on()
    ax2.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
    ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
    ax2.minorticks_on()
    
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
        
    # Set limits
    if False == True:
        ax1.set_xlim(0, 11.8)
        ax1.set_ylim(0, 1.5e-3)
        ax2.set_ylim(0, 4.2)
        
    # Configure ticks spacing
    if False == True:
        ax1.set_xticks(np.arange(0, 12, 2))
        #ax1.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
        ax1.set_yticks(np.arange(0, 1.51e-3, 0.2e-3))
        #ax1.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)
        ax2.set_yticks(np.arange(0, 4.21, 0.5))
        #ax2.set_yticks(np.arange(0, 1.01, 0.02), minor=True)
        
    handles_list = [
        mlines.Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label=r"$p(r)$"),
        mlines.Line2D([], [], color=colors[0], linestyle='--', linewidth=1.5, label=r"$m(r)$"),
        ]
        
    plt.legend(handles=handles_list, fontsize=15, loc = "upper right", bbox_to_anchor=(0.99, 0.99), frameon=True, fancybox=False,
               ncol = 2,edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
        
    plt.tight_layout()
    plt.savefig(rf"figures\poly_TOV.pdf", format="pdf", bbox_inches="tight")
    plt.show()

