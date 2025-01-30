# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:47:59 2025

Code That writes test eos

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

def create_eos(Type, p_range, n, plot=False):
    """
    Function that generates an excel containing the data of the equation of state

    Parameters
    ----------
    Type : string
        Type of eos that you require. For a constant density star use 'constant', for a polytropic eos use 'polytropic'.
    p_range : touple
        Vector that contains the initial and final preassures. The structure is as follows: (pi, pf).
    n : interger
        Number of data points you want the data to have.
    plot : Boolian
        Wether or not the code should make a plot of the created eos.

    Returns
    -------
    None.

    """
    
    pi, pf = p_range
    
    if Type == "constant":
        p = np.linspace(pi, pf, n)
        rho = np.zeros(n) + 2.3873241463784e-4
            
    elif Type == "polytropic":
        K=150
        gamma=2
        p = np.linspace(pi, pf, n)
        rho = (p / K) ** (1 / gamma)
    
    data = pd.DataFrame({'Density': rho, 'Pressure': p})
    data.to_excel("data.xlsx", index=False)
    
    if plot==True:
        plt.figure(figsize=(8, 8))
        plt.plot(p, rho, label = r'$p(\rho)$', color = "black", linewidth = 2, linestyle = '-', marker = '.', markersize = 10)
        #plt.title(r'Created eos', loc='center', fontsize=20, fontweight='bold')
        plt.xlabel(r'$p$ $\left[ M_{\odot}/km^3 \right]$', fontsize=15, loc='center', fontweight='bold')
        plt.ylabel(r'$\rho$ $\left[ M_{\odot}/km^3 \right]$', fontsize=15, loc='center', fontweight='bold')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=12, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=6, width=1.2, labelsize=10, top=True, right=True)
        plt.minorticks_on()
        plt.gca().spines['top'].set_linewidth(1.7)
        plt.gca().spines['right'].set_linewidth(1.7)
        plt.gca().spines['bottom'].set_linewidth(1.7)
        plt.gca().spines['left'].set_linewidth(1.7)
        plt.legend(fontsize=17, frameon=False)
        plt.show()
        
    return None

create_eos("polytropic", (1e-9, 1e5), 20, True)
