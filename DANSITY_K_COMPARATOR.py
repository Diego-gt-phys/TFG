# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 15:01:46 2025

Creates a plot that compares the star by varying only the k parameter.

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

###############################################################################
# Read the data
###############################################################################

K_range = (5, 10, 15, 20, 25, 35)
TYPE, EOS, ALPHA, pc = ("TOV", "soft", 0.2, 1e-5)
data = {}

for K in K_range:
    df = pd.read_csv(f"data_{TYPE}_{EOS}_{ALPHA}_{pc}_{K}.csv")
    data[K] = df
    
###############################################################################
# Plot the data
###############################################################################

plt.figure(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)
styles = ((0,()), (0,(3,1.6)), (0,(6,1.6,1,1.6)), (0,(1,1.6)), (0,(3,1.6,1,1.6)), (0,(6,1.6)))

# Scale factors
p_scale = 1e5
m_scale = 1
# EOS color
if EOS == "soft": 
    c=0 #
elif EOS == 'middle':
    c=1 
elif EOS == 'stiff':
    c=2 
i=0
for K in K_range:
    plt.plot(data[K]["r"], data[K]["m"]*m_scale, label = rf'$K={K}$', color = 'k', linewidth = 1.5, linestyle = styles[i])
    plt.plot(data[K]["r"], data[K]["p_B"]*p_scale, label = '', color = colors[6], linewidth = 1.5, linestyle = styles[i])
    plt.plot(data[K]["r"], data[K]["m_A"]*m_scale, label = '', color = colors[c], linewidth = 1.5, linestyle = styles[i])
    plt.plot(data[K]["r"], data[K]["m_B"]*m_scale, label = '', color = colors[3], linewidth = 1.5, linestyle = styles[i])    
    plt.axvline(data[K]["R_A"][0], color=colors[8], linewidth=1.0, linestyle=styles[i])
    i+=1

###############################################################################
# Configure the plot
###############################################################################

# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
plt.title(rf'Comparison of TOV solution for different $K$ and {EOS} EoS', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
plt.ylabel(r'$p\cdot 10^5$ $\left[ M_{\odot}/km^3\right]$ & $m$ $\left[ M_{\odot}\right]$', fontsize=15, loc='center')
plt.axhline(0, color='k', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='k', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
plt.xlim(0, 33.82)
plt.ylim(0, 0.5)

# Add grid
#plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
#plt.gca().set_xticks(np.arange(0, 12.1, 1))  # Major x ticks 
#plt.gca().set_yticks(np.arange(0, 8.1, 1))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=12, frameon=False, ncol = 1) #  loc='upper right',

# Save the plot as a PDF
plt.savefig(f"Fig_K_COMP_{EOS}_{ALPHA}_{pc}.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()

