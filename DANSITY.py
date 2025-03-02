# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:59:52 2025

DANSITI

Solves the TOV equation for Dark matter (fluid B) Admixed Neutron Star (fluis A)

@author: Diego García Tejada
"""


###############################################################################
# Imports and units
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

# Physical parameters (solar mass = 198847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses
PCS = {"soft": (2.785e-6, 5.975e-4), "middle": (2.747e-6, 5.713e-4), "stiff": (2.144e-6, 2.802e-4)} # Central pressure intervals for the MR curves 

###############################################################################
# Define the functions
###############################################################################

def eos_A (p_A): # The fluid A is the Neutron matter
    """
    Guiven the arrays p_data and rho_data which contain the information for the equation of state, this funtion interpolates the value of rho for a guiven  p. 

    Parameters
    ----------
    p_A : float
        Preassure of fluid A at which we want to evaluate the eos.

    Returns
    -------
    rho : float
        Density associated to the preassure guiven.

    """
    if p_A <= 0:
        return 0
    
    interp_func = interp1d(p_data, rho_data, kind='linear', fill_value='extrapolate')
    rho = interp_func(p_A)
    
    return rho

def eos_B (p_B): # The fluid B is the dark matter.
    """
    Polytropic EOS for the star B.
    
    Parameters:
        p_B (float): Preassure of the fluid B.
        
    Returns:
        float: Density of the fluid B (rho) at preassure p_B.
    """

    if p_B <= 0:
        return 0  # Avoid invalid values
        
    rho = (p_B / 35) ** (3 / 5) # 1e23
    return rho

def system_of_ODE (r, y):
    """
    Function that calculates the derivatives of m and the pressures. This function is used for the runge-Kutta method. 

    Parameters
    ----------
    r : float
        radius inside of the star.
    y : tuple
        (m, p_A, p_B), where m is the mass, p_A[p_B] is the preassure of the fluid [B], evaluated at point r.

    Returns
    -------
    dm_dr : float
        rate of change of the mass.
    dpA_dr : float
        rate of change of the pressure of fluid A.
    dpB_dr : float
        rate of change of the preassure of fluid B.
    """
    
    m, p_A, p_B, m_A, m_B = y
    rho_A = eos_A(p_A)
    rho_B = eos_B(p_B)
    
    dm_dr = 4 * np.pi * (rho_A + rho_B) * r**2
    
    dmA_dr = 4 * np.pi * (rho_A) * r**2
    
    dmB_dr = 4 * np.pi * (rho_B) * r**2
    
    dphi_dr = (G * m + 4 * np.pi * G * r**3 * (p_A + p_B)) / (r * (r - 2 * G * m))
    
    dpA_dr = -(rho_A + p_A) * dphi_dr
    
    dpB_dr = -(rho_B + p_B) * dphi_dr
    
    return (dm_dr, dpA_dr, dpB_dr, dmA_dr, dmB_dr)

def RK4O_with_stop (y0, r_range, h):
    """
    Function that integrates the y vector using a Runge-Kutta 4th orther method.
    Due to the physics of our problem. The function is built with a condition that doesn't allow negative pressures. If both of them are 0 then the integration stops. 

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_A_c, P_B_c, m_A_0, m_B_0)
    r_range : tuple
        Range of integratio: (r_0, r_max)
    h : float
        Step size of integration.

    Returns
    -------
    r_values : array
        Array containing the different values of r.
        
    y_values : array
        Array containig the solutions for the vector y: (m_values, p_A_values, P_B_values).
        
    R_A : float
        Value of the radius of the star of fluid A
    """
    
    r_start, r_end = r_range
    r_values = [r_start]
    y_values = [y0]
    
    r = r_start
    y = np.array(y0)
    
    R_A = 0

    while r <= r_end:
        k1 = h * np.array(system_of_ODE(r, y))
        k2 = h * np.array(system_of_ODE(r + h / 2, y + k1 / 2))
        k3 = h * np.array(system_of_ODE(r + h / 2, y + k2 / 2))
        k4 = h * np.array(system_of_ODE(r + h, y + k3))

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Stopping conditions: p<p_c*1e-10
        if y_next[1] <= y0[1]*1e-10 and y_next[2] <= y0[2]*1e-10: # If both pressures drop to 0 then the star stops there.
            break
        
        elif y_next[1] <= y0[1]*1e-10 and R_A == 0:  # If fluid A's pressure drops to 0, keep it that way
            y_next[1] = 0
            R_A = r
            print("The star has a Halo of Dark Matter")
            
        elif y_next[1] <= y0[1]*1e-10 and R_A != 0:  # If fluid A's pressure drops to 0, keep it that way
            y_next[1] = 0
            
        elif y_next[2] <= y0[2]*1e-10:  # If fluid B's pressure drops to 0, keep it that way
            y_next[2] = 0

        r += h
        r_values.append(r)
        y_values.append(y_next)
        y = y_next
        
    if R_A == 0:
        R_A = r_values[-1]
        print("The star has a core of Dark Matter")
        
    return (np.array(r_values), np.array(y_values), R_A)

def TOV_solver(y0, r_range, h):
    """
    Using a 4th Runge Kutta method, it solves the TOV for a 2 perfect fluid star.
    It guives the mass and preassure values for both fluids in all of the stars radius.

    Parameters
    ----------
    y0 : tuple
        Starting conditions for our variables: (m_0, p_A_c, P_B_c, m_A_0, m_B_0)
    r_range : tuple
        Range of integration: (r_0, r_max)
    h : float
        Step size of integration.

    Returns
    -------
    r_values : array
        Array containing the different values of r.
    m_values : array
        Array containing the different values of m(r).
    p_A_values : array
        Array containing the different values of P_A(r).
    p_B_values : array
        Array containing the different values of P_B(r).
    R_A : float
        Value of the radius of the star of fluid A
    """
    
    r_values, y_values, R_A = RK4O_with_stop(y0, r_range, h)
    
    m_values = y_values[:, 0]
    p_A_values = y_values[:, 1]
    p_B_values = y_values[:, 2]
    m_A_values = y_values[:, 3]
    m_B_values = y_values[:, 4]

    return (r_values, m_values, p_A_values, p_B_values, m_A_values, m_B_values, R_A)

def MR_curve(pc_range, alpha, r_range, h, n):
    """
    Creates the mass radius curve of a family of 2-fluid stars by solving the TOV equations.
    The different masses are calculating by changing the central pressure of fluid A and mantaining the alpha value of fluid B. 

    Parameters
    ----------
    pc_range : tuple
        Range of central pressures to integrate: (pc_start, pc_end)
    alpha : float
        Relationship between the central pressure of fluid B and fluid A: pc_B = alpha * pc_A
    r_range : tuple
        Range of radius integration: (r_0, r_max)
    h : float
        integration step.
    n : int
        Number of datapoints of the curve.

    Returns
    -------
    R_values : array
        Array containig the radius of the stars.
    M_values : array
        Array containing the total masses of the stars.
    MA_values : array
        Array containing the mass of fluid A of the stars.
    MB_values : array
        Array containing the mass of fluid B of the stars.
    """
    pc_start, pc_end = pc_range
    pc_list = np.geomspace(pc_start, pc_end, n) 
    R_values = []
    M_values = []
    MA_values = []
    MB_values = []
    cont = 0
    for pc in pc_list:
        r_i, m_i, p_A, p_B, m_a, m_b, R_A = TOV_solver((0, pc, alpha*pc, 0, 0), r_range, h)
        
        R_i = R_A
        M_i = m_i[-1]
        MA_i = m_a[-1]
        MB_i = m_b[-1]
        
        R_values.append(R_i)
        M_values.append(M_i)
        MA_values.append(MA_i)
        MB_values.append(MB_i)
        
        cont += 1
        print(cont)
    
    return (np.array(R_values), np.array(M_values), np.array(MA_values), np.array(MB_values))

###############################################################################
# Define the parameters
###############################################################################

CHOICE, TYPE, EOS, ALPHA, PC = (1, "MR", "soft", 0.1, 3e-4)

###############################################################################
# Create the data
###############################################################################
if CHOICE == 0:
    eos_data = pd.read_excel(f"eos_{EOS}.xlsx")
    rho_data = eos_data['Density'].values
    p_data = eos_data['Pressure'].values
    data = {}

    if TYPE == "TOV":
        # Calculate
        r, m, p_A, p_B, m_A, m_B, R_A = TOV_solver((0, PC, ALPHA*PC, 0, 0), (1e-6, 50), 1e-3)
        # Insert in dict
        data["r"] = r
        data["m"] = m
        data["p_A"] = p_A
        data["p_B"] = p_B
        data["m_A"] = m_A
        data["m_B"] = m_B
        data["R_A"] = R_A
        # Save the data
        df = pd.DataFrame(data)
        df.to_csv(f"data_{TYPE}_{EOS}_{ALPHA}_{PC}.csv", index=False)
        print("Data saved.")
        
    elif TYPE == "MR":
        pc_range = PCS[f"{EOS}"]
        # Calculate
        R, M, M_A, M_B = MR_curve(pc_range, ALPHA, (1e-6, 50), 1e-3, 20)
        # Insert data
        data["R"] = R
        data["M"] = M
        data["M_A"] = M_A
        data["M_B"] = M_B
        df = pd.DataFrame(data)
        df.to_csv(f"data_{TYPE}_{EOS}_{ALPHA}.csv", index=False)
        print("Data saved.")
###############################################################################
# Plot the data
###############################################################################
elif CHOICE == 1:
    
    if TYPE == "TOV":
        # Read the data
        df = pd.read_csv(f"data_{TYPE}_{EOS}_{ALPHA}_{PC}.csv")
        r = df["r"]
        p_A = df["p_A"]
        p_B = df["p_B"]
        m = df["m"]
        m_A = df["m_A"]
        m_B = df["m_B"]
        
        # Scale factors
        p_scale = 1e4
        m_scale = 1
        
        # Configure the plot
        plt.figure(figsize=(9.71, 6))
        colors = sns.color_palette("Set1", 10)
        
        # Plot the data
        plt.plot(r, p_A*p_scale, label = r'$p_{soft}(r) \cdot 10^4$', color = colors[0], linewidth = 1.5, linestyle = '-') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m_A*m_scale, label = r'$m_{soft}(r)$', color = colors[0], linewidth = 1.5, linestyle = '-.') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, p_B*p_scale, label = r'$p_{DM}(r) \cdot 10^4$', color = colors[3], linewidth = 1.5, linestyle = '-') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m_B*m_scale, label = r'$m_{DM}(r)$', color = colors[3], linewidth = 1.5, linestyle = '-.') # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(r, m*m_scale, label = r'$m(r) / 3$', color = 'k', linewidth = 1.5, linestyle = '--') # , marker = "*",  mfc='w', mec = 'w', ms = 5

        # Set the axis to logarithmic scale
        #plt.xscale('log')
        #plt.yscale('log')
        
        # Add labels and title
        plt.title(rf'TOV solution for the {EOS} eos and $\alpha = {ALPHA}$', loc='left', fontsize=15, fontweight='bold')
        plt.xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
        plt.ylabel(r'$p\cdot 10^4$ $\left[ M_{\odot}/km^3\right]$ & $m$ $\left[ M_{\odot}\right]$', fontsize=15, loc='center')
        plt.axhline(0, color='k', linewidth=1.0, linestyle='--')  # x-axis
        plt.axvline(0, color='k', linewidth=1.0, linestyle='--')  # y-axis
        
        # Set limits
        plt.xlim(0, 10.02)
        plt.ylim(0, 3)

        # Add grid
        #plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Configure ticks for all four sides
        plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
        plt.minorticks_on()

        # Customize tick spacing for more frequent ticks on x-axis
        plt.gca().set_xticks(np.arange(0, 10.56, 1))  # Major x ticks 
        plt.gca().set_yticks(np.arange(0, 3.01, 0.5))  # Major y ticks 

        # Set thicker axes
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)

        # Add a legend
        plt.legend(fontsize=12, frameon=False, ncol = 3) #  loc='upper right',

        # Save the plot as a PDF
        plt.savefig(f"fig_{TYPE}_{EOS}_{ALPHA}_{PC}.pdf", format="pdf", bbox_inches="tight")

        plt.tight_layout()
        plt.show()
        
    elif TYPE == "MR":
        # Read the data
        df = pd.read_csv(f"data_{TYPE}_{EOS}_{ALPHA}.csv")
        R = df["R"]
        M = df["M"]
        M_A = df["M_A"]
        M_B = df["M_B"]
        
        # Configure the plot
        plt.figure(figsize=(9.71, 6))
        colors = sns.color_palette("Set1", 10)
        
        # Plot the data
        plt.plot(R, M, label = r'$M(R)$', color = colors[0], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5) # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(R, M_A, label = r'$M_{NS}(R)$', color = colors[1], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5) # , marker = "*",  mfc='w', mec = 'w', ms = 5
        plt.plot(R, M_B, label = r'$M_{DM}(R)$', color = colors[2], linewidth = 1.5, linestyle = '-', marker = "*",  mfc='k', mec = 'k', ms = 5) # , marker = "*",  mfc='w', mec = 'w', ms = 5
        

        # Add labels and title
        plt.title(rf'MR curve for the {EOS} eos and $\alpha = {ALPHA}$', loc='left', fontsize=15, fontweight='bold')
        plt.xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
        plt.ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center')
        
        # Set limits
        plt.xlim(8, 17)
        plt.ylim(0, 3.5)
        
        # Add grid
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

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
        plt.legend(fontsize=12, frameon=False, ncol = 1) #  loc='upper right',

        # Save the plot as a PDF
        plt.savefig(f"fig_{TYPE}_{EOS}_{ALPHA}.pdf", format="pdf", bbox_inches="tight")

        plt.tight_layout()
        plt.show()
        