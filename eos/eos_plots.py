# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:44:54 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

G = 1.4765679173556 # G in units of km / solar masses

# Read the data
soft = pd.read_excel("fixed_soft.xlsx")
soft_density = soft['x'].values / G
soft_preassure = soft['y'].values / G
middle = pd.read_excel("fixed_middle.xlsx")
middle_density = middle['x'].values / G
middle_preassure = middle['y'].values / G
stiff = pd.read_excel("fixed_stiff.xlsx")
stiff_density = stiff['x'].values / G
stiff_preassure = stiff['y'].values / G

# Plot the data

plt.figure(figsize=(9, 6.94))
colors = sns.color_palette("Set1", 5)
plt.plot(soft_density, soft_preassure, label = r'soft', color = colors[0], linewidth = 2, linestyle = '-', marker = "o",  mfc=colors[0], mec = 'k', ms = 5)
plt.plot(middle_density, middle_preassure, label = r'middle', color = colors[1], linewidth = 2, linestyle = '-', marker = "s",  mfc=colors[1], mec = 'k', ms = 5)
plt.plot(stiff_density, stiff_preassure, label = r'stiff', color = colors[2], linewidth = 2, linestyle = '-', marker = "D",  mfc=colors[2], mec = 'k', ms = 5)

plt.title(r'Equations of state', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$\mathbf{\rho} \cdot 10^3$ $\left[M_{\odot}/km^{3}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'$\mathbf{p} \cdot 10^4$ $\left[M_{\odot}/km^{3}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis
plt.xlim(0, 0.83)
plt.ylim(0, 2.71)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()
plt.gca().set_xticks(np.arange(0, 0.81, 0.1))  # Major x ticks 
plt.gca().set_yticks(np.arange(0, 2.51, 0.5))  # Major y ticks 
plt.gca().spines['top'].set_linewidth(1.6)
plt.gca().spines['right'].set_linewidth(1.6)
plt.gca().spines['bottom'].set_linewidth(1.6)
plt.gca().spines['left'].set_linewidth(1.6)
plt.legend(fontsize=15, frameon=False)

plt.savefig("eos_with_markers.pdf", format="pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()
