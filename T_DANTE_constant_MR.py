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

def theoretical_data (M):
    """
    Solves the hydrostatic equillibrium of a star of constant density and mass M ussing the theoretical equations.

    Parameters
    ----------
    M : float
        Mass of the star.

    Returns
    -------
    r_teo : array
        Array containing the theoretical values of r.
    m_teo : TYPE
        Array containing the theoretical values of m(r).
    p_teo : TYPE
        Array containing the theoretical values of p(r).
    """
    rho = 5e-4
    R = (3/4 * M/(rho*np.pi))**(1/3)
    
    r_teo = np.linspace(0, R, 500)
    m_teo = 4/3 * rho * np.pi * r_teo**3
    term1 = np.sqrt(R**3 - 2 * G * M * R**2)
    term2 = np.sqrt(R**3 - 2 * G * M * r_teo**2)
    numerator = term1 - term2
    denominator = term2 - 3 * term1
    p_teo = rho * (numerator / denominator)
    
    return (r_teo, m_teo, p_teo)
      
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

    R, M = MR_curve((1e-2, 1e-10), (1e-6, 100), 1e-3, 100)

###############################################################################
# Plot the data
###############################################################################
    
    fig, ax = plt.subplots(figsize=(9.71, 6))
    colors = sns.color_palette("Set1", 10)
    
    # Plot
    ax.plot(R, M, label=r'M(R)', color = colors[0], linewidth=1.5, linestyle='-', marker = "*",  mfc='k', mec = 'k', ms = 5)
    ax.set_xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
    ax.set_ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center', color='k')
    ax.tick_params(axis='y', colors='k')
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    
    # Add axis lines
    ax.axhline(0, color='k', linewidth=1.0, linestyle='--')
    ax.axvline(0, color='k', linewidth=1.0, linestyle='--')
    
    # Configure ticks
    ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
    ax.minorticks_on()
    
    # Set thicker axes
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
        
    # Set limits
    if True == True:
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        
    # Configure ticks spacing
    if True == True:
        ax.set_xticks(np.arange(0, 50.1, 5))
        #ax.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
        ax.set_yticks(np.arange(0, 1.001, 0.2))
        #ax.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)
        
        
    plt.legend(fontsize=15, loc = "upper right", bbox_to_anchor=(0.994, 0.99), frameon=True, fancybox=False,
               ncol = 3,edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)
        
    plt.tight_layout()
    plt.savefig(rf"figures\poly_MR.pdf", format="pdf", bbox_inches="tight")
    plt.show()


