# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 09:42:20 2025
 
In this code ill try to plot the complete Equation of State (EoS) of a State of a Completely Degenerate, Ideal Fermi Gas.
I'll follow the equations in Saphiro's Black Holes, White Dwarfs and Neutron Stars: The Physics of Compact Objects.

WARNING: The notation in the book is different. My stress-energy tensor works with energy density (epsilon), not mass density (rho_0).

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import seaborn as sns

# I belive that numerically it fails at scales of x = 10^-10

x = np.geomspace(1e-3, 10, 5000)

def phi(x):
    return np.sqrt(1 + x**2) * ((2/3) * x**3 - x) + np.log(x + np.sqrt(1 + x**2))

def psi(x):
    return np.sqrt(1 + x**2) * (2 * x**3 + x) - np.log(x + np.sqrt(1 + x**2))

def phi_9th(x):
    return (8/15) * x**5 - (4/21) * x**7 + (1/9) * x**9

def psi_9th(x):
    return (8/3) * x**3 + (4/5) * x**5 - (1/7) * x**7 + (1/18) * x**9

def phi_nr(x):
    return (8/15) * x**5

def psi_nr(x):
    return (8/3) * x**3

###############################################################################
# Test of the code
###############################################################################

p = 4.7038237577116e-7 * phi(x)
rho = 4.7038237577116e-7 * psi(x)
p_9th = 4.7038237577116e-7 * phi_9th(x)
rho_9th = 4.7038237577116e-7 * psi_9th(x)
p_nr = 4.7038237577116e-7 * phi_nr(x)
rho_nr = 4.7038237577116e-7 * psi_nr(x)

data = {0:[rho,p], 1:[rho_9th, p_9th], 2:[rho_nr, p_nr]}

# Configure the plot
fig, ax1 = plt.subplots(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 11)
styles = ['-', '--', '-.']
s=0

# Configure the axis
ax1.set_xlabel(r'$\xi$', fontsize=15, loc='center')
ax1.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2 = ax1.twinx()
ax2.set_ylabel(r'$\rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax2.tick_params(axis='y', colors='k')
ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax2.set_yscale('log')

# Plot
for key in data.keys():
    ax1.plot(x, data[key][1], color=colors[0], label=rf'$p_{{{key}}}$', linewidth=1.5, linestyle=styles[s]) #, marker = "+",  mfc='k', mec = 'k', ms = 5
    ax2.plot(x, data[key][0], color=colors[1], label=rf'$\rho_{{{key}}}$', linewidth=1.5, linestyle=styles[s])
    s += 1
    
# Set limits
ax1.set_xlim(1e-2, 1e1)
ax1.set_ylim(1e-18, 1e0)
ax2.set_ylim(1e-18, 1e0)

# Configure ticks
ax1.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True)
ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True)
ax1.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
ax2.minorticks_on()

# Configure ticks spacing
#ax1.set_xticks(np.arange(0, 2.51, 0.25))
#ax1.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
ax1.set_yticks(np.geomspace(1e-18, 1e0, 10))
#ax1.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)
ax2.set_yticks(np.geomspace(1e-16, 1e0, 9))
#ax2.set_yticks(np.arange(0, 1.01, 0.02), minor=True)

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

# Make legend
handles_list = [
    mlines.Line2D([], [], color=colors[0], linestyle='-', label=r"$p(\xi)$"),
    mlines.Line2D([], [], color=colors[1], linestyle='-', label=r"$\rho(\xi)$"),
    mlines.Line2D([], [], color='k', linestyle='-', label=r"Relativistic"),
    mlines.Line2D([], [], color='k', linestyle='--', label=r"Quasi-Relativistic"),
    mlines.Line2D([], [], color='k', linestyle='-.', label=r"Non-Relativistic")]

plt.legend(handles = handles_list, loc = "upper left", bbox_to_anchor=(0.009, 0.99), fontsize=15, frameon=True, fancybox=False, ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

plt.tight_layout()
plt.savefig("Parametrization_test_log.pdf", format="pdf", bbox_inches="tight")

plt.show()
