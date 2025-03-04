# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 22:27:20 2025

Plots the three MR curves for the diferent eos

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

alphas = (0, 0.05, 0.1)
EsOS = ("soft", "middle", "stiff")
data = {}

for alpha in alphas:
    for EOS in EsOS:
        df = pd.read_csv(f"data_MR_{EOS}_{alpha}.csv")
        data[f"{EOS}_{alpha}"] = df

###############################################################################
# Plot the data
###############################################################################

plt.figure(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)

i=0
styles = ("-", "-.", "--")
for EOS in EsOS:
    j=0
    for alpha in alphas:
        plt.plot(data[f"{EOS}_{alpha}"]["R"], data[f"{EOS}_{alpha}"]["M"], label = rf'{EOS} $({alpha})$', color = colors[i], linewidth = 1.5, linestyle = styles[j], marker = "*",  mfc='k', mec = 'k', ms = 5)
        j += 1
    i += 1

###############################################################################
# Configure the plot
###############################################################################

# Add labels and title
plt.title(r'MR curves for $\alpha = (0, 0.05, 0.1)$', loc='left', fontsize=15, fontweight='bold')
plt.xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
plt.ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center')

# Set limits
plt.xlim(8, 17)
plt.ylim(0, 3.5)

# Add grid
#plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

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
plt.legend(fontsize=12, frameon=False, ncol = 3, loc = 'upper left') #  loc='upper right',

# Save the plot as a PDF
plt.savefig("FIG_MR_complete.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()  
        
        