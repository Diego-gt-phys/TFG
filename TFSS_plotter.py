# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:56:30 2025

Plots the three MR curves for the diferent eos

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

# Alpha parameter
ALPHA = 0

# Read the data
df_soft_0 = pd.read_csv(f"MR_soft_{ALPHA}.csv")
df_middle_0 = pd.read_csv(f"MR_middle_{ALPHA}.csv")
df_stiff_0 = pd.read_csv(f"MR_stiff_{ALPHA}.csv")

# Alpha parameter
ALPHA = 0.05

# Read the data
df_soft_0_05 = pd.read_csv(f"MR_soft_{ALPHA}.csv")
df_middle_0_05 = pd.read_csv(f"MR_middle_{ALPHA}.csv")
df_stiff_0_05 = pd.read_csv(f"MR_stiff_{ALPHA}.csv")

# Alpha parameter
ALPHA = 0.1

# Read the data
df_soft_1 = pd.read_csv(f"MR_soft_{ALPHA}.csv")
df_middle_1 = pd.read_csv(f"MR_middle_{ALPHA}.csv")
df_stiff_1 = pd.read_csv(f"MR_stiff_{ALPHA}.csv")

# Configure the plot
plt.figure(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)

# Plot the data
plt.plot(df_soft_0["R"], df_soft_0["M"], label = r'soft $(.0)$', color = colors[0], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
plt.plot(df_soft_0_05["R"], df_soft_0_05["M"], label = r'soft $(.05)$', color = colors[0], linewidth = 1.5, linestyle = '-.', marker = "*",  mfc='k', mec = 'k', ms = 5)
plt.plot(df_soft_1["R"], df_soft_1["M"], label = r'soft $(.1)$', color = colors[0], linewidth = 1.5, linestyle = '--', marker = "*",  mfc='k', mec = 'k', ms = 5)

plt.plot(df_middle_0["R"], df_middle_0["M"], label = r'middle $(.0)$', color = colors[1], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
plt.plot(df_middle_0_05["R"], df_middle_0_05["M"], label = r'middle $(.05)$', color = colors[1], linewidth = 1.5, linestyle = '-.', marker = "*",  mfc='k', mec = 'k', ms = 5)
plt.plot(df_middle_1["R"], df_middle_1["M"], label = r'middle $(.1)$', color = colors[1], linewidth = 1.5, linestyle = '--', marker = "*",  mfc='k', mec = 'k', ms = 5)

plt.plot(df_stiff_0["R"], df_stiff_0["M"], label = r'stiff $(.0)$', color = colors[2], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5)
plt.plot(df_stiff_0_05["R"], df_stiff_0_05["M"], label = r'stiff $(.05)$', color = colors[2], linewidth = 1.5, linestyle = '-.', marker = "*",  mfc='k', mec = 'k', ms = 5)
plt.plot(df_stiff_1["R"], df_stiff_1["M"], label = r'stiff $(.1)$', color = colors[2], linewidth = 1.5, linestyle = '--', marker = "*",  mfc='k', mec = 'k', ms = 5)

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
plt.legend(fontsize=12, frameon=False, ncol = 1, loc = 'upper right') #  loc='upper right',

# Save the plot as a PDF
plt.savefig(f"MR_all_EOS_multy_alpha.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()