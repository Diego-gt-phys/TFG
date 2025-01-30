# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:42:41 2025

In this code I'll make a interpolation function that takes an excel and is able to interpolate the data.

@author: Usuario
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Constants
K = 150
gamma = 2

# Generate 20 points for density (rho) in a reasonable range
p_values = np.linspace(1e-9, 1e5, 10)  # Adjust range as needed
rho_values = (p_values / K) ** (1 / gamma)

# Create a DataFrame
data = pd.DataFrame({'Density': rho_values, 'Pressure': p_values})
data.to_excel("data.xlsx", index=False)

# Read data from Excel using NumPy for speed
data_read = pd.read_excel("data.xlsx")
rho_read = data_read['Density'].values
p_read = data_read['Pressure'].values

# Interpolation function
def eos(p):
    interp_func = interp1d(p_read, rho_read, kind='linear', fill_value='extrapolate')
    return interp_func(p)

print("La densidad para una presión de 40000 es: ", eos(40000))