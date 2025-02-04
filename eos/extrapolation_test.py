# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:24:36 2025

This code tests the extrapolation capabilities at very low values

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

# Read the data
data = pd.read_excel("soft.xlsx")
rho_data = data['Density'].values
p_data = data['Pressure'].values

# Interpolate the data. For fun we'll call it the signal
def find_signal(p):
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p)
    
    return rho

p = np.linspace(0, 2.7e-4, 10000)
rho = find_signal(p)

plt.figure(figsize=(9, 6.94))
colors = sns.color_palette("Set1", 5)
plt.plot(rho, p, label = r'Signal', color = colors[0], linewidth = 2, linestyle = '-', marker = "",  mfc=colors[0], mec = colors[0], ms = 5)
plt.plot(rho_data, p_data, 'o', label = r'Data', color = 'k', ms = 3)

plt.title(r'Soft equation of state', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$\mathbf{\rho}$ $\left[M_{\odot}/km^{3}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.ylabel(r'$\mathbf{p}$ $\left[M_{\odot}/km^{3}\right]$', fontsize=15, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(0, 1e-4)
#plt.ylim(0, 1e-6)
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
plt.legend(fontsize=15, frameon=False)

plt.savefig("soft_eos.pdf", format="pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()