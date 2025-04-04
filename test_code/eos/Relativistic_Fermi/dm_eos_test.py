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

x = np.geomspace(1e-4, 10, 5000)

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

def plytropic(rho):
    K = 8.0164772576091
    return K*rho**(5/3)

###############################################################################
# Test of the code
###############################################################################

p = 1.4777498161008e-3 * phi(x)
rho = 1.4777498161008e-3 * psi(x)
p_9th = 1.4777498161008e-3 * phi_9th(x)
rho_9th = 1.4777498161008e-3 * psi_9th(x)
p_nr = 1.4777498161008e-3 * phi_nr(x)
rho_nr = 1.4777498161008e-3 * psi_nr(x)
rho_ply = np.geomspace(1e-12, 1e2)
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

# Plot
ax.plot(rho, p, label=r'EoS', color=colors[0], linewidth=1.5, linestyle='-')
ax.plot(rho_ply, p_ply, label=r'NR-EoS', color=colors[1], linewidth=1.5, linestyle='--')
ax.plot(rho_9th, p_9th, label=r'QR-EoS', color=colors[2], linewidth=1.5, linestyle='-.')


    
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
plt.savefig("DIFG_EoS.pdf", format="pdf", bbox_inches="tight")

plt.show()
