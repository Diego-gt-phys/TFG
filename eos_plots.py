# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:44:54 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read the data
soft = pd.read_excel("soft_cleaned.xlsx")
soft_density = soft['x'].values
soft_preassure = soft['y'].values
middle = pd.read_excel("middle_cleaned.xlsx")
middle_density = middle['x'].values
middle_preassure = middle['y'].values 

# Plot the data

plt.figure(figsize=(9, 6.94))
colors = sns.color_palette("Set1", 5)
plt.plot(soft_density, soft_preassure, label = r'soft', color = colors[0], linewidth = 2, linestyle = '-', marker = "o",  mfc=colors[0], mec = 'k', ms = 5)
plt.plot(middle_density, middle_preassure, label = r'middle', color = colors[1], linewidth = 2, linestyle = '-', marker = "s",  mfc=colors[1], mec = 'k', ms = 5)

plt.title(r'Equations of state', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$\mathbf{\rho} \cdot 10^3$ $[km^{-2}]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'$\mathbf{p} \cdot 10^4$ $[km^{-2}]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis
#plt.xlim(0, 1.2e-3)
#plt.ylim(0, 4e-4)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()
plt.gca().set_xticks(np.arange(0, 1.201, 0.1))  # Major x ticks 
plt.gca().set_yticks(np.arange(0, 4.01, 0.5))  # Major y ticks 
plt.gca().spines['top'].set_linewidth(1.6)
plt.gca().spines['right'].set_linewidth(1.6)
plt.gca().spines['bottom'].set_linewidth(1.6)
plt.gca().spines['left'].set_linewidth(1.6)
plt.legend(fontsize=15, frameon=False, loc='upper left')

plt.savefig("eos.pdf", format="pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()
