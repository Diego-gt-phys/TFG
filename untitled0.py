# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:31:08 2025

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define x range
x_values = np.linspace(0.1, 10, 100)  # Avoiding zero to prevent issues with exponentiation

# Define exponent values
a_values = np.arange(0, 1.1, 0.1)

# Create a dictionary to store data
data = {"x": x_values}

# Compute y values for each a
for a in a_values:
    data[f"y_a={a:.1f}"] = x_values ** a

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("power_function_data.csv", index=False)

print("Data saved to 'power_function_data.csv'")

# Load the data from CSV
df_loaded = pd.read_csv("power_function_data.csv")

# Plot the data
plt.figure(figsize=(8, 6))
for column in df_loaded.columns[1:]:  # Skip the first column (x values)
    a_value = column.split("=")[1]  # Extract the exponent value
    plt.plot(df_loaded["x"], df_loaded[column], label=fr"$y = x^{{{a_value}}}$")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Plot of $y = x^a$ for different values of $a$")
plt.legend()
plt.grid(True)
plt.show()
