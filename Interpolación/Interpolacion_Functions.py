# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:47:59 2025

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

def create_eos(type, p_range, n):
    """
    Function that generates an excel containing the data of the equation of state

    Parameters
    ----------
    type : string
        Type of eos that you require. For a constant density star use 'constant', for a polytropic eos use 'polytropic'.
    p_range : touple
        Vector that contains the initial and final preassures. The structure is as follows: (pi, pf).
    n : interger
        Number of data points you want the data to have.

    Returns
    -------
    None.

    """
    
    pi, pf = p_range
    
    if type == "constant":
        p = np.linspace(pi, pf, n)
        rho = np.zeros(n) + 2.3873241463784e-4
            
    elif type == "polytropic":
        K=150
        gamma=2
        p = np.linspace(pi, pf, n)
        rho = (p / K) ** (1 / gamma)
    
    data = pd.DataFrame({'Density': rho, 'Pressure': p})
    data.to_excel("data.xlsx", index=False)
    
    return None

    print ("test")