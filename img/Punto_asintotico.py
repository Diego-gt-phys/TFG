# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:53:30 2024

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

t = np.logspace(np.log10(0.1),20, 100)
x = 5-1/(t+0.2)
y = x**4

x_teo = np.linspace(0, 5, 500)
y_teo = x_teo**4

# Start plotting procedure
colors = sns.color_palette("Set1", 5)

plt.figure(figsize=(9.71, 6)) # The image follows the golden ratio
plt.plot(x,y, label = r'$x(t)=5 -\frac{1}{t+0.2}$ & $y(x)=x^2$', color = colors[0], linewidth = 1.5, linestyle = '-', marker = '+', markersize=10)
plt.plot(x_teo,y_teo, label = r'Curva teórica', color = 'black', linewidth = 1, linestyle = '-', marker = '', markersize=10)


# Set the axis to logarithmic scale
#plt.xscale('log')
#plt.yscale('log')

# Add labels and title
plt.title(r'Curva con punto asintótico', loc='left', fontsize=14, fontweight='bold')
plt.xlabel(r'x', fontsize=12, loc='center', fontweight='bold')
plt.ylabel(r'y', fontsize=12, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
plt.xlim(0,5.2)
plt.ylim(0,660)

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
plt.gca().set_xticks(np.arange(0, 5.21, 0.5))  # Major x ticks 
plt.gca().set_yticks(np.arange(100, 660, 100))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=15, frameon=False) #  loc='upper right',

# Save the plot as a PDF
#plt.savefig("Punto_asintotico_con_bucle.pdf.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()