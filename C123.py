from prepare import *
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


df_in = pd.read_csv('full_df')
pd.set_option('display.max_columns', None)

def equations(vars, w20, w90, w120):
    C1, C2, C3 = vars
    eq1 = C1 + C2*np.sin(np.deg2rad(20)) + C3*np.cos(np.deg2rad(20)) - w20
    eq2 = C1 + C2*np.sin(np.deg2rad(90)) + C3*np.cos(np.deg2rad(90)) - w90
    eq3 = C1 + C2*np.sin(np.deg2rad(120)) + C3*np.cos(np.deg2rad(120)) - w120
    return [eq1, eq2, eq3]

def solve_row(row):
    # Initial guess for Cs
    initial_guess = (0, 0, 0)
    w20, w90, w120 = row['tauspinner_HCP_Theta_20'], row['tauspinner_HCP_Theta_90'], row['tauspinner_HCP_Theta_120']
    solution = fsolve(equations, initial_guess, args=(w20, w90, w120))

    return solution

solutions = df_in.apply(solve_row, axis=1)
print(len(solutions))

# Convert solutions to a DataFrame and merge with the original DataFrame
solutions_df = pd.DataFrame(solutions.tolist(), columns=['C1', 'C2', 'C3'])
df_in = df_in.join(solutions_df)

print(df_in.head(10))

plt.figure(figsize=(10, 6))
plt.hist(df_in['C1'], bins=200, color='blue', edgecolor='black')
plt.title('C1')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_in['C2'], bins=200, color='blue', edgecolor='black')
plt.title('C2')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_in['C3'], bins=200, color='blue', edgecolor='black')
plt.title('C3')
plt.show()
