import time
import csv
import inspect
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
from matplotlib.animation import FuncAnimation, PillowWriter


# number of modes, for HH Hamiltonian, we have x, y two modes
n = 2


# Henon-Heiles system Hamiltonian
lam = 0.09
h_variables = [1/2, 1/2, lam, - lam / 3]
H = henon_heiles(h_variables)

# Occupation number (Number of basis functions)
truncation = 60
truncations = [60, 70]
energy_list = []
for truncation in truncations:

    # Calculate the exact eigenvalues and eigenvectors w.r.t. the truncation
    eigenvalues1, e1 = boson_eigenspectrum_full(H, truncation)
    ee1 = e1[:, 0]
    f1 = ee1.reshape(truncation, truncation)

    unbound_point = float(unbound_criterion(alpha=lam))
    print(f"Unbound point: {unbound_point}")

    # List to store parameters for all states
    parameters_list = []

    # range of eigenvalues and state to calculate
    ranges = [20 * x for x in range(9)]

    energies = []

    # Loop over all the eigenstates
    for i in range(200):

        ee2 = e1[:, i]  # Extract i-th eigenstate
        ff1 = ee2.reshape(truncation, truncation)
        if i == 0:
            print(eigenvalues1[0])
        energies.append(eigenvalues1[i])

    energy_list.append(np.array(energies))
    # energy_list.append(energies)

differences = abs(energy_list[1] - energy_list[0])


truncation = 60
# Calculate the exact eigenvalues and eigenvectors w.r.t. the truncation
eigenvalues1, e1 = boson_eigenspectrum_full(H, truncation)
ee1 = e1[:, 0]
f1 = ee1.reshape(truncation, truncation)

unbound_point = float(unbound_criterion(alpha=lam))
print(f"Unbound point: {unbound_point}")

# List to store parameters for all states
parameters_list = []

# range of eigenvalues and state to calculate
values = [20 * x for x in range(9)]
# Loop over all the eigenstates
for i in range(200):
    ee2 = e1[:, i]  # Extract i-th eigenstate
    ff1 = ee2.reshape(truncation, truncation)

    MI_evolved = []
    # t_list = [100 * x for x in range(1, 5)]
    # for t in t_list:
    #     time_evolved = get_time_evolution(H, truncation, ee2, t)
    #     # ff1_evolved = time_evolved.reshape(truncation, truncation)
    #     ff1_evolved = np.outer(time_evolved, time_evolved)
    #     print(ff1_evolved.shape)
    #     MI_evolved.append(von_neumann_entropy_rdm(ff1_evolved, truncation))

    # print(f"Time evolved MI: {MI_evolved}")
    # Append the parameters for the current state
    parameters_list.append([f'State {i}',
        round(eigenvalues1[i], 7),
        round(mutual_information(ff1, truncation), 7),
        round(covariance(ee2, truncation), 7),
        round(abs(covariance_x2(ee2, truncation)), 7),
        round(covariance_x21(ee2, truncation), 7)
    ])


energies = np.array(parameters_list)[:,1].astype(float)
mutual_info = np.abs(np.array(parameters_list)[:,2].astype(float))
# plt.plot(energies, mutual_info, 'o', label='Energy vs Mutual Info')
covar_x = np.array(parameters_list)[:,3].astype(float)
covar_x2 = np.array(parameters_list)[:,4].astype(float)
covar_x21 = np.abs(np.array(parameters_list)[:,5].astype(float))

# Print results for all states
# for i, params in enumerate(parameters_list):
#         print(f'{params}')


crit_E = critical_energy(lam)
# energies_below_crit = energies[energies < crit_E]
# covar_x2_below_crit = mutual_info[energies < crit_E]
#
# # Fit for data points below critical energy
# params_below_crit, covariance_below_crit = curve_fit(model_func, energies_below_crit, covar_x2_below_crit)
# fit_below_crit = model_func(energies_below_crit, *params_below_crit)
#
# # Fit for all data points
# params_all, covariance_all = curve_fit(model_func, energies, covar_x2)
# fit_all = model_func(energies, *params_all)
#
# # Calculate chi-square for both fits
# chi_square_below_crit = np.sum((covar_x2_below_crit - fit_below_crit) ** 2 / fit_below_crit)
# chi_square_all = np.sum((mutual_info - fit_all) ** 2 / fit_all)

# Plotting
# plt.plot(energies, mutual_info, 'o', label=f'Energy vsMI ($\lambda = {lam}$)')
# plt.plot(energies_below_crit, fit_below_crit, 'r-',
#          label=f'Fit below Crit Energy (χ²={chi_square_below_crit:.2f})')
# plt.plot(energies, fit_all, 'b-',
#          label=f'Fit for all data (χ²={chi_square_all:.2f})')

plt.xlabel('Energy')
plt.ylabel('| Covariance $x^2 y^2$ |')
print(type(unbound_point), unbound_point)
plt.axvline(x=np.round(unbound_point, 2),
            color='r', linestyle='--', label=f'Unbound border: {np.round(unbound_point, 2)}')

plt.axvline(x=np.round(crit_E, 2), color='g', linestyle='--', label=f'Critical Energy: {np.round(crit_E, 2)}')

high_diff = differences > 1E-6
low_diff = ~high_diff

# Create color labels based on the condition
# colors = ['red' if diff > 1E-3 else 'blue' for diff in differences]
# plt.plot(energies, mutual_info, label='Mutual Information')

# Plot points with high differences
plt.scatter(energies[high_diff], covar_x2[high_diff], c='red', label='Non-converging points')
#
# Plot points with low differences
plt.scatter(energies[low_diff], covar_x2[low_diff], c='blue', label='Converging points')
plt.title("$H = (p_x^2 + x^2) / 2 + (p_y^2 + y^2) / 2 + \lambda (x^2y - y^3/3) \lambda = 0.09$")
plt.legend()
plt.show()


# print("Parameters (Below Critical Energy):", params_below_crit)
# print("Chi-square (Below Critical Energy):", chi_square_below_crit)
#
# print("Parameters (All Data Points):", params_all)
# print("Chi-square (All Data Points):", chi_square_all)
# save_parameters(parameters_list,lam,truncation)
