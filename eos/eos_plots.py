# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:44:54 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

"""
def find_signal(p):
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p)
    
    return rho
"""

# Read the data
soft = pd.read_excel("soft.xlsx")
soft_density = soft['Density'].values
soft_preassure = soft['Pressure'].values
middle = pd.read_excel("middle.xlsx")
middle_density = middle['Density'].values
middle_preassure = middle['Pressure'].values
stiff = pd.read_excel("stiff.xlsx")
stiff_density = stiff['Density'].values
stiff_preassure = stiff['Pressure'].values

# Interpolate the data
p_soft = np.geomspace(5.65e-31, 6e-4, 10000)
eos_soft = interp1d(soft_preassure, soft_density, kind='linear', fill_value='extrapolate')
rho_soft = eos_soft(p_soft)
p_middle = np.geomspace(5.65e-31, 5.7e-4, 10000)
eos_middle = interp1d(middle_preassure, middle_density, kind='linear', fill_value='extrapolate')
rho_middle = eos_middle(p_middle)
p_stiff = np.geomspace(5.65e-31, 2.8e-4, 10000)
eos_stiff = interp1d(stiff_preassure, stiff_density, kind='linear', fill_value='extrapolate')
rho_stiff = eos_stiff(p_stiff)

# Plot the data
plt.figure(figsize=(9, 6.94))
colors = sns.color_palette("Set1", 5)

#plt.plot(rho_soft, p_soft, label = r'fitted: soft', color = colors[0], linewidth = 2, linestyle = '-', marker = "",  mfc=colors[0], mec = 'k', ms = 5)
plt.plot(soft_density, soft_preassure, label = r'soft', color = colors[0], linewidth = 2, linestyle = '', marker = "o",  mfc=colors[0], mec = 'k', ms = 5)
#plt.plot(rho_middle, p_middle, label = r'fitted: middle', color = colors[1], linewidth = 2, linestyle = '-', marker = "",  mfc=colors[1], mec = 'k', ms = 5)
plt.plot(middle_density, middle_preassure, label = r'middle', color = colors[1], linewidth = 2, linestyle = '', marker = "s",  mfc=colors[1], mec = 'k', ms = 5)
#plt.plot(rho_stiff, p_stiff, label = r'fitted: siff', color = colors[2], linewidth = 2, linestyle = '-', marker = "",  mfc=colors[2], mec = 'k', ms = 5)
plt.plot(stiff_density, stiff_preassure, label = r'stiff', color = colors[2], linewidth = 2, linestyle = '', marker = "D",  mfc=colors[2], mec = 'k', ms = 5)


plt.title(r'Interpolated fit for the EsOS', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$\mathbf{\rho}$ $\left[M_{\odot}/km^{3}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'$\mathbf{p}$ $\left[M_{\odot}/km^{3}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(0, 9.5e-4)
plt.ylim(0, 6e-4)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()
#plt.gca().set_xticks(np.arange(0, 0.81, 0.1))  # Major x ticks 
#plt.gca().set_yticks(np.arange(0, 2.51, 0.5))  # Major y ticks 
plt.gca().spines['top'].set_linewidth(1.6)
plt.gca().spines['right'].set_linewidth(1.6)
plt.gca().spines['bottom'].set_linewidth(1.6)
plt.gca().spines['left'].set_linewidth(1.6)
plt.legend(fontsize=15, frameon=True, framealpha=0.9, edgecolor='k', loc = 'upper left')

plt.savefig("eos_data.pdf", format="pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()
