# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:12:34 2025

Plots the TOV solution for the same Pc and different alpha

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

###############################################################################
# Read the data
###############################################################################

pc = 3e-6
alphas = ("0", "0.05", "0.1")
data = {}

for alpha in alphas:
    df = pd.read_csv(f"data_TOV_soft_{alpha}_{pc}.csv")
    data[f"{alpha}"] = df

###############################################################################
# Plot the data
###############################################################################

plt.figure(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)

# Scale factors
p_scale = 1e5
m_scale = 1

i = 0
styles = ("-", "-.", "--")
for alpha in alphas:
    plt.plot(data[alpha]["r"], data[alpha]["p_A"]*p_scale, label = rf'$p_{{NS}}({alpha}) \cdot 10^5$', color = colors[i], linewidth = 1.5, linestyle = '-')
    plt.plot(data[alpha]["r"], data[alpha]["m"]*m_scale, label = rf'$m({alpha})$', color = colors[i], linewidth = 1.5, linestyle = '-.')
    plt.plot(data[alpha]["r"], data[alpha]["m_A"]*m_scale, label = rf'$m_{{NS}}({alpha})$', color = colors[i], linewidth = 1.5, linestyle = '--')
    plt.plot(data[alpha]["r"], data[alpha]["m_B"]*m_scale, label = rf'$m_{{DM}}({alpha})$', color = colors[3], linewidth = 1.5, linestyle = styles[i])
    i += 1

###############################################################################
# Configure the plot
###############################################################################

# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
plt.title(rf'TOV solution for the soft eos and $\alpha = (0, 0.05, 0.1)$', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
plt.ylabel(r'$p\cdot 10^5$ $\left[ M_{\odot}/km^3\right]$ & $m$ $\left[ M_{\odot}\right]$', fontsize=15, loc='center')
plt.axhline(0, color='k', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='k', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
plt.xlim(0, 16)
plt.ylim(0, 0.3)

# Add grid
#plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
#plt.gca().set_xticks(np.arange(0, 22.16, 2))  # Major x ticks 
plt.gca().set_yticks(np.arange(0, 0.301, 0.05))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=12, frameon=False, ncol = 3) #  loc='upper right',

# Save the plot as a PDF
plt.savefig(f"FIG_alpha_{pc}.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()
