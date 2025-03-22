# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 21:53:41 2025

Graphs MR curves for data calculted previusly using DANTE.py

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

###############################################################################
# Read the data
###############################################################################

data = {}

DM_mass = 1
EsOS = ["soft", "middle", "stiff"]
param_vals = [0.0, 0.02, 0.05]

for eos_c in EsOS:
    for param_val in param_vals:
        df = pd.read_csv(rf"..\data\4_{eos_c}_l_{param_val}_{DM_mass}.csv")
        data[f"{eos_c}_{param_val}"] = df

###############################################################################
# Plot the data
###############################################################################

plt.figure(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)

styles = ["-", "--", "-."]

c = 0
for eos_c in EsOS:
    s = 0
    for param_val in param_vals:
        plt.plot(data[f"{eos_c}_{param_val}"]["R"], data[f"{eos_c}_{param_val}"]["M"], label = r'', color = colors[c], linewidth = 1.5, linestyle = styles[s], marker = "*",  mfc='k', mec = 'k', ms = 5)
        s+=1
    c+=1

# Manually fake the legend
plt.plot([-2,-1], [-2,-1], label = r'$\lambda = 0$', color = 'k', linewidth = 1.5, linestyle = '-')
plt.plot([-2,-1], [-2,-1], label = r'$\lambda = 0.02$', color = 'k', linewidth = 1.5, linestyle = '--')
plt.plot([-2,-1], [-2,-1], label = r'$\lambda = 0.05$', color = 'k', linewidth = 1.5, linestyle = '-.')


###############################################################################
# Configure the plot
###############################################################################
# Add labels and title
plt.title(r'MR curves of DANS: $m_{{\chi}}=1[GeV]$', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
plt.ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center')

# Set limits
plt.xlim(8, 17)
plt.ylim(0, 3.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
plt.gca().set_xticks(np.arange(8, 17.1, 1))  # Major x ticks 
plt.gca().set_yticks(np.arange(0, 3.51, 0.5))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=12, frameon=False, ncol = 1, loc = 'upper left') #  loc='upper right',

# Save the plot as a PDF
plt.savefig(f"MR_curves_DANS_{DM_mass}.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()