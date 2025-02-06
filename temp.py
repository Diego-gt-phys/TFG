# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 00:08:57 2025

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

def eos(p):
    """
    Guiven the arrays p_data and rho_data which contain the information for the equation of state, this funtion interpolates the value of rho for a guiven  p. 

    Parameters
    ----------
    p : float
        Preassure at which we want to evaluate the eos.

    Returns
    -------
    rho : float
        Density associated to the preassure guiven.

    """
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p)
    
    return rho

data = pd.read_excel("stiff.xlsx")
rho_data = data['Density'].values
p_data = data['Pressure'].values

print(eos(1e-20))
