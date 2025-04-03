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
import matplotlib.ticker as ticker
import seaborn as sns

def eos_B (p_B):
    # We work with an Ideal Fermi Gass
    Gamma = 5/3 # Polytropic coeficient for a degenetare (T=0) IFG
    K = ((m)**(-8/3))*8.0165485819726 # Polytropic constant for a degenetare (T=0) IFG

    #if p_B <= 0:
        #return 0  # Avoid invalid values
        
    rho = (p_B / K) ** (1/Gamma)
    return rho

def phi(x):
    return (1 / (8 * np.pi**2)) * (x * np.sqrt(1 + x**2) * ((2/3) * x**2 - 1) + np.log(x + np.sqrt(1 + x**2)))

def chi(x):
    return (1 / (8 * np.pi**2)) * (x * np.sqrt(1 + x**2) * (1 + 2*x**2) - np.log(x + np.sqrt(1 + x**2)))

def alpha(m):
    return m**4 * 4.7038237577114e-4

m = 1
x = np.logspace(-3.3, 1, 100)

y_1 = phi(x)
y_2 = chi(x)

p = alpha(m) * phi(x)
rho = alpha(m) * chi(x)

p_DANTE = np.logspace(-20, -1, 100)
rho_DANTE = eos_B(p_DANTE)

plt.figure(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)

plt.plot(rho, p, label=r'EoS', color=colors[0], linewidth=1.5, linestyle='-')
plt.plot(rho_DANTE, p_DANTE, label=r'DANTE', color=colors[1], linewidth=1.5, linestyle='--')


plt.title(r'Equation of State of a Completely Degenerate, Ideal Fermi Gas', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$\rho$ $\left[M_{\odot}/km^3\right]$', fontsize=15, loc='center')
plt.ylabel(r'$p$ $\left[ M_{\odot}/km^3 \right]$', fontsize=15, loc='center')
plt.axvline(0, color='k', linewidth=1.0, linestyle='--')
plt.axhline(0, color='k', linewidth=1.0, linestyle='--')


# Aplly scientific notation to y axis
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))

# Set log scale for both axes
#plt.xscale('log')
#plt.yscale('log')

# Set limits
#plt.xlim(1e-15, 1e0)
#plt.ylim(1e-20, 1e0)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing 
#plt.gca().set_xticks(np.arange(8, 17.1, 1))  # Major x ticks 
#plt.gca().set_yticks(np.arange(0, 3.51, 0.5))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

plt.legend(fontsize=15, frameon=True, fancybox=False, ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

# Save plot as PDF
plt.tight_layout()
plt.savefig("dm_eos_test.pdf", format="pdf", bbox_inches="tight")

plt.show()
