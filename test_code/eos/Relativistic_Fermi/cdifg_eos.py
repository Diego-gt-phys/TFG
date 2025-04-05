# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 22:03:40 2025

Code used for creating the dataset conaining the information of the EoS of a Completely Degenerate Ideal Fermi Gas,
whose particles have a guiven mass in the order of GeV.

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

    def phi(x):
        return np.sqrt(1 + x**2) * ((2/3) * x**3 - x) + np.log(x + np.sqrt(1 + x**2))

    def psi(x):
        return np.sqrt(1 + x**2) * (2 * x**3 + x) - np.log(x + np.sqrt(1 + x**2))
    
    def phi_21st(x):
        return (8/15) * x**5 - (4/21) * x**7 + (1/9) * x**9 - (5/66) * x**11 + (35/624) * x**13 - (7/160) * x**15 + (77/2176) * x**17 - (143/4864) * x**19 + (715/28672) * x**21

    def psi_21st(x):
        return (8/3) * x**3 + (4/5) * x**5 - (1/7) * x**7 + (1/18) * x**9 - (5/176) * x**11 + (7/416) * x**13 - (7/640) * x**15 + (33/4352) * x**17 - (429/77824) * x**19 + (715/172032) * x**21

    k = (dm_m)**(4) * 1.4777498161008e-3
    
    x_vals = np.geomspace(1e-6, 10, 5000)

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
    print("Data saved")
    
calc_CDIFG(1)

df = pd.read_excel('eos_data.xlsx')
rho = df['rho']
p = df['P']

x = np.geomspace(1e-6, 10, 5000)

def phi(x):
    return np.sqrt(1 + x**2) * ((2/3) * x**3 - x) + np.log(x + np.sqrt(1 + x**2))

def psi(x):
    return np.sqrt(1 + x**2) * (2 * x**3 + x) - np.log(x + np.sqrt(1 + x**2))

p_teo = 1.4777498161008e-3 * phi(x)
rho_teo = 1.4777498161008e-3 * psi(x)

def plytropic(rho):
    K = 8.0164772576091
    return K*rho**(5/3)

rho_ply = np.geomspace(1e-15, 1e2)
p_ply = plytropic(rho_ply)

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

ax.plot(rho, p, label=r'Calculated EoS', color=colors[0], linewidth=1.5, linestyle='-')
ax.plot(rho_teo, p_teo, label=r'Theoretical EoS', color=colors[1], linewidth=1, linestyle='-.')
ax.plot(rho_ply, p_ply, label=r'Non relativistic EoS', color=colors[2], linewidth=1, linestyle='--')

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
plt.savefig("DIFG_EoS_calculated.pdf", format="pdf", bbox_inches="tight")

plt.show()
