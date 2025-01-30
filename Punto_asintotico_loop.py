# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:24:35 2024

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openpyxl import load_workbook  # Required for appending to Excel

"""
# Function to calculate x(t) and y(t) for a given t interval
def calculate_parametric_data(t_start, t_end, num_points):
    t = np.logspace(t_start, t_end, num_points)
    x = 5 - 1 / (t + 0.2)
    y = x**4
    return x, y

# Parameters for intervals
t_start = -5  # Initial starting time
t_end = 20  # Final time
step = 1     # Interval size
num_points = 10  # Points per interval

# Excel file setup
excel_file = "parametric_curve_data.xlsx"

# Loop over intervals and append data
for t_i in range(t_start, t_end, step):
    t_min = t_i
    t_max = t_i + step
    x, y = calculate_parametric_data(t_min, t_max, num_points)

    # Save data to Excel
    df_parametric = pd.DataFrame({'x': x, 'y': y})
    
    try:
        # Append data to the existing sheet
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Start appending after the last row of the sheet
            df_parametric.to_excel(writer, index=False, header=False, sheet_name='Sheet1', 
                                   startrow=writer.sheets['Sheet1'].max_row)
    except FileNotFoundError:
        # Create a new file if it doesn't exist
        df_parametric.to_excel(excel_file, index=False)

"""
# Generate theoretical curve
x_teo = np.linspace(0, 5, 500)
y_teo = x_teo**4

# Extract data
df = pd.read_excel("parametric_curve_data.xlsx")
x = df["x"]
y = df["y"]

#Plot
plt.figure(figsize=(9.71, 6)) # The image follows the golden ratio
colors = sns.color_palette("Set1", 5)
plt.plot(x,y, label = r'$x(t)=5 -\frac{1}{t+0.2}$ & $y(x)=x^2$', color = colors[0], linewidth = 1.5, linestyle = '-', marker = '+', markersize=5)
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
#plt.savefig("Punto_asintotico_con_bucle.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()