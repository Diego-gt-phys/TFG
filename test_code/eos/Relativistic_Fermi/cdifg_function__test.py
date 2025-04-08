# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:37:30 2025

Test of the two functions before adding them into Dante.py.

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d # Needed for interpolation of EoS
import scipy.optimize as opt # Needed to find the values od lambda

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
    
    x_vals = np.geomspace(1e-6, 10, 1000)

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

    df.to_excel(f'eos_cdifg_{dm_m}.xlsx', index=False)
    
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
        eos_data = pd.read_excel(f'eos_cdifg_{dm_m}.xlsx')
        rho_data_dm = eos_data['rho'].values
        p_data_dm = eos_data['P'].values
    except FileNotFoundError:
        print("File not found, calculating new file.")
        eos_data = calc_CDIFG(dm_m)
        rho_data_dm = eos_data['rho'].values
        p_data_dm = eos_data['P'].values
        
    return p_data_dm, rho_data_dm

def eos_B (p_B):
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
    
    interp_func = interp1d(p_data_dm, rho_data_dm, kind='linear', fill_value='extrapolate')
    rho = interp_func(p_B)
    
    return rho

p_data_dm, rho_data_dm = read_create_dm_eos(1)

p_list = np.geomspace(1e-18, 1e2, 1000)
p = []
rho = []

for p_B in p_list:
    p.append(p_B)
    rho_B = eos_B(p_B)
    rho.append(rho_B)
    
# Configure the plot
fig, ax = plt.subplots(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 11)


# Configure the axis
ax.set_xlabel(r'$\rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax.tick_params(axis='y', colors='k')
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(rho, p, label=r'Calculated EoS', color=colors[0], linewidth=1.5, linestyle='-', marker = ".",  mfc='k', mec = 'k', ms = 5)
ax.plot(rho_data_dm, p_data_dm, label=r'Theoretical EoS', color=colors[1], linewidth=1, linestyle='-.')

# Set limits
ax.set_xlim(1e-11, 2e1)
ax.set_ylim(1e-17, 2e1)


# Configure ticks
ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
ax.minorticks_on()

# Configure ticks spacing
ax.set_xticks(np.geomspace(1e-11, 1e1, 13))
#ax.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
ax.set_yticks(np.geomspace(1e-17, 1e1, 10))
#ax.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)


# Set thicker axes
for ax in [ax]:
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')

# Make legend
plt.legend(loc = "upper left", bbox_to_anchor=(0.009, 0.99), fontsize=15, frameon=True, fancybox=False, ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

# Save fig as pdf
plt.title(r'EoS of a Completely Degenerate, Ideal Fermi Gas', loc='left', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("Function_test.pdf", format="pdf", bbox_inches="tight")

plt.show()