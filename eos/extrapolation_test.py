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
data = pd.read_excel("middle.xlsx")
rho_data = data['Density'].values
p_data = data['Pressure'].values

# Interpolate the data. For fun we'll call it the signal
def find_signal(p):
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p)
    
    return rho

p = np.linspace(start, stop)