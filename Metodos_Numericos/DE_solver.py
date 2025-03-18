# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:39:45 2024

@author: Diego García Tejada
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.ioff() #DEBUG Si no haces esto, el código no guarda la imagen.

# Write the differential equation to solve
def dydx(x,y):
    return y

# Non optimized Runge-Kutta method
def runge_kutta_old(x0, y0, h, xf):
    n = int((xf - x0) / h)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0
    for i in range(1, n + 1):
        k1 = h * dydx(x[i - 1], y[i - 1])
        k2 = h * dydx(x[i - 1] + h / 2, y[i - 1] + k1 / 2)
        k3 = h * dydx(x[i - 1] + h / 2, y[i - 1] + k2 / 2)
        k4 = h * dydx(x[i - 1] + h, y[i - 1] + k3)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[i] = x[i - 1] + h
    return x, y

# Optimized Runge-Kutta 4th order method
def runge_kutta(x0, y0, h, xf):
    n = int((xf - x0) / h) # Number of iterations
    
    # Initialize arrays for x and y
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0] = x0
    y[0] = y0

    # Precompute h/2 to avoid redundant calculation
    h_half = h / 2
    dydx_vals = np.empty(4)  # Array to store dydx evaluations

    for i in range(1, n + 1):
        x_i_minus_1 = x[i - 1] 
        y_i_minus_1 = y[i - 1]
        
        # Precompute dydx values and store them in dydx_vals
        dydx_vals[0] = dydx(x_i_minus_1, y_i_minus_1)  # k1
        dydx_vals[1] = dydx(x_i_minus_1 + h_half, y_i_minus_1 + h_half * dydx_vals[0])  # k2
        dydx_vals[2] = dydx(x_i_minus_1 + h_half, y_i_minus_1 + h_half * dydx_vals[1])  # k3
        dydx_vals[3] = dydx(x_i_minus_1 + h, y_i_minus_1 + h*dydx_vals[2])  # k4
        
        # Compute y[i] using precomputed dydx values
        y[i] = y_i_minus_1 + (h / 6) * (dydx_vals[0] + 2 * dydx_vals[1] + 2 * dydx_vals[2] + dydx_vals[3])
        x[i] = x_i_minus_1 + h
    return x, y

# Write the initial conditions
x0 = 0
y0 = 1
h = 0.005
xf = 10

# Solve the DE
x,y = runge_kutta(x0, y0, h, xf)
#x_old, y_old = runge_kutta_old(x0, y0, h, xf)

# Calculate the exact solution
x_teo = np.linspace(x0, xf, 500)
y_teo = np.exp(x_teo)

# Generate a color palette
colors = sns.color_palette("Set1", 5)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'Método optimizado', color=colors[0], linewidth=2) # Numerical solution
#plt.plot(x_old, y_old, label=r'Método antiguo', color=colors[1], linewidth=2, linestyle = '-.') 
plt.plot(x_teo, y_teo, label=r'Solución analítica', color="black", linewidth=1, linestyle = '--') # Theoretical solution

# Add labels and title
plt.title(r'Resolución de Ecuaciones Diferenciales', loc='left', fontsize=14, fontweight='bold')
plt.xlabel('x', fontsize=12, loc='center', fontweight='bold')
plt.ylabel('y', fontsize=12, loc='center', fontweight='bold')
plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # x-axis
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # y-axis

# Set limits
#plt.xlim(x0,xf)
#plt.ylim(0,20)

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=9, width=1.3, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=5, width=1.0, labelsize=10, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing for more frequent ticks on x-axis
#plt.gca().set_xticks(np.arange(x0, xf + 0.01, 1))  # Major x ticks 
#plt.gca().set_yticks(np.arange(0, 1 + 0.01, 0.2))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.2)
plt.gca().spines['right'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['left'].set_linewidth(1.2)

# Add a legend
plt.legend(fontsize=12, loc='upper left', frameon=False)

# Save the plot as a PDF
#plt.savefig("Figura_1.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.tight_layout()
plt.show()