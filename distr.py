# dystrybucje phi star dla wag roznych hipotez

from prepare import *
import matplotlib.pyplot as plt
import pandas as pd

# zero = Events('nazwy2.txt.root')
# zero.to_csv('pik_zero.csv')
###
zero = pd.read_csv('pik_zero.csv')
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