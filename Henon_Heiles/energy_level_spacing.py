import numpy as np
from Henon_Heiles.critical_energy import unbound_criterion
from utils.util_hamil import henon_heiles, test,test1,test2,test5

from utils.util_mutualinfo import mutual_information, von_neumann_entropy_rdm
from utils.util_covar import covariance,covariance_x2,covariance_x21
from ten_network import mps1,cpd1,mps2,cpd2
from utils.util_save import printx,save_parameters
from utils.util_gfro import (obtain_fragment,
                       rotated_hamiltonian,
                       boson_eigenspectrum_sparse,
                       get_time_evolution,
                       boson_eigenspectrum_full,
                       quad_diagonalization)
from Henon_Heiles.critical_energy import critical_energy

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


n = 2


# Henon-Heiles system Hamiltonian
lam = 0.09
h_variables = [1/2, 1/2, lam, - lam / 3]
H = henon_heiles(h_variables)


truncation = 60
# Calculate the exact eigenvalues and eigenvectors w.r.t. the truncation
eigenvalues1, e1 = boson_eigenspectrum_full(H, truncation)
ee1 = e1[:, 0]
f1 = ee1.reshape(truncation, truncation)

unbound_point = float(unbound_criterion(alpha=lam))
print(f"Unbound point: {unbound_point}")

# List to store parameters for all states
eigen_energies = []
energy_spacings = []

crit_E = critical_energy(lam)

# # range of eigenvalues and state to calculate
# # Loop over all the eigenstates
for i in range(200):
    ev = eigenvalues1[i]  # Extract i-th eigenstate
    eig_1 = eigenvalues1[i]
    eig_2 = eigenvalues1[i+1]
    energy_diff = eig_2 - eig_1
    eigen_energies.append(eig_1)
    energy_spacings.append(energy_diff)


# unique_energies = np.unique(eigenvalues1)
# print(f"Length of unique energies: {len(unique_energies)}")
# # Ensure eigenvalues are sorted
# energy_levels = np.sort(unique_energies[:4])
#
# # Compute energy level spacings
# energy_spacings = np.diff(energy_levels)
# print(energy_spacings[:5])
#
# # Convert to a flat NumPy array to avoid interpretation as multiple datasets
# energy_spacings = np.asarray(energy_spacings).flatten()

# Plot histogram of level spacings
plt.figure(figsize=(8, 5))
# energy_spacings = (energy_spacings - np.min(energy_spacings)) / (np.max(energy_spacings) - np.min(energy_spacings))
# plt.hist(energy_spacings, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=np.round(104, 2), color='g', linestyle='--', label=f'Critical Energy: {np.round(crit_E, 2)}')
plt.scatter([i for i in range(len(energy_spacings))], energy_spacings, c='red', label='Non-converging points')

plt.xlabel("Energy indices")
plt.ylabel("Energy spacing")
plt.title("Distribution of Energy Level Spacings")
plt.grid(True)
plt.show()
