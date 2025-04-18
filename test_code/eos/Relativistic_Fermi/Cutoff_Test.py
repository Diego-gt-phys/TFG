# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:53:42 2025

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d # Needed for interpolation of EoS
import scipy.optimize as opt # Needed to find roots

def eos_B (p_B, cutoff = 1e-20): # DM
    """
    For a dm_m, guiven the arrays 'p_data_dm' and 'rho_data_d,' which contain the information for the equation of state for the dark matter,
    this funtion interpolates the value of rho for a guiven  p. 

    Parameters
    ----------
    p_B : float
        Preassure of fluid B at which we want to evaluate the eos.
        
    cutoff : float
        Value for wich the eos uses the nonrelativistic aproximation.

    Returns
    -------
    rho : float
        Density associated to the preassure guiven.
    """
    if p_B <= 0:
        return 0
    
    elif p_B <= cutoff:
        Gamma = 5/3
        K = ((dm_m)**(-8/3))*8.0165485819726
        rho = (p_B / K) ** (1/Gamma)
    
    else:
        interp_func = interp1d(p_data_dm, rho_data_dm, kind='linear', fill_value='extrapolate')
        rho = interp_func(p_B)
    
    return rho

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
        eos_data = calc_CDIFG(dm_m)
        rho_data_dm = eos_data['rho'].values
        p_data_dm = eos_data['P'].values
        
    return p_data_dm, rho_data_dm

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
        
    x_vals = np.geomspace(1e-6, 10, 10000)
    
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
    
    df.to_excel(rf'eos_cdifg_{dm_m}.xlsx', index=False)
        
    return df

###############################################################################

dm_m = 1
p_data_dm, rho_data_dm = read_create_dm_eos(dm_m)

p = np.geomspace(1e-10, 1e-3, 1000)
rho_theo = []
rho_nr = []

for p_i in p:
    
    rho_theo_i = eos_B(p_i)
    rho_nr_i = eos_B(p_i, cutoff=1e-2)

    rho_theo.append(rho_theo_i)
    rho_nr.append(rho_nr_i)
    
res = np.array(rho_theo) - np.array(rho_nr)

fig, (ax_main, ax_resid) = plt.subplots(1, 2, sharey=True, figsize=(9.71, 6), gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0})
colors = sns.color_palette("Set1", 11)
    
ax_main.plot(rho_theo, p, label=r'R', color=colors[0], linewidth=1.5, linestyle='-')
ax_main.plot(rho_nr, p, label=r'NR', color=colors[1], linewidth=1.5, linestyle='-.')
ax_resid.plot(res, p, label=r'RESD', color='k', linewidth=1.5, linestyle='-')

ax_main.set_xlabel(r'$\rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax_main.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax_main.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_main.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
ax_main.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_main.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
ax_main.set_xscale('log')
ax_main.set_yscale('log')

ax_resid.set_xlabel(r'$\Delta \rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax_resid.set_xscale('log')

ax_main.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=False)
ax_main.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=False)
ax_main.minorticks_on()
ax_resid.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True, left=False)
ax_resid.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True, left=False)
ax_resid.minorticks_on()

# Set thicker axes
for ax in [ax_main, ax_resid]:
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')

plt.tight_layout()
plt.savefig("Cutoff_test.pdf", format="pdf", bbox_inches="tight")

plt.show()