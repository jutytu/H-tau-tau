# Plots distributions of phi star for different hypotheses.


import matplotlib.pyplot as plt
import pandas as pd

def Plots(data)
zero = pd.read_csv(data) # this data frame is called 'zero' due to the name of the first file it was tested on
plt.subplot(1, 2, 1)
plt.hist(zero['phi_star'], bins=50, weights=zero['tauspinner_HCP_Theta_160'])
plt.xlabel('Phi_Star')
plt.ylabel('Counts')
plt.title('160')
###
plt.subplot(1, 2, 2)
plt.hist(zero['phi_star'], bins=50, weights=zero['tauspinner_HCP_Theta_40'])
plt.xlabel('Phi_Star')
plt.ylabel('Counts')
plt.title('40')
plt.show()

nz = pd.read_csv('df3.csv')
plt.subplot(1, 2, 1)
plt.hist(nz['phi_star'], bins=50, weights=nz['tauspinner_HCP_Theta_160'])
plt.xlabel('Phi_Star')
plt.ylabel('Counts')
plt.title('160')
###
plt.subplot(1, 2, 2)
plt.hist(nz['phi_star'], bins=50, weights=nz['tauspinner_HCP_Theta_40'])
plt.xlabel('Phi_Star')
plt.ylabel('Counts')
plt.title('40')
plt.show()

plt.subplot(2, 2, 1)
plt.hist(nz['phi_star'], bins=50, weights=nz['tauspinner_HCP_Theta_0'])
plt.title('0')
###
plt.subplot(2, 2, 2)
plt.hist(nz['phi_star'], bins=50, weights=nz['tauspinner_HCP_Theta_40'])
plt.title('40')
###
plt.subplot(2, 2, 3)
plt.hist(nz['phi_star'], bins=50, weights=nz['tauspinner_HCP_Theta_90'])
plt.title('90')
###
plt.subplot(2, 2, 4)
plt.hist(nz['phi_star'], bins=50, weights=nz['tauspinner_HCP_Theta_140'])
plt.title('140')
###
plt.show()
