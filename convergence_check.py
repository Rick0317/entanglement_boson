import numpy as np
from Henon_Heiles.critical_energy import unbound_criterion
from utils.util_hamil import henon_heiles, test,test1,test2,test5

from utils.util_mutualinfo import mutual_information
from utils.util_covar import covariance,covariance_x2,covariance_x21
from ten_network import mps1,cpd1,mps2,cpd2
from utils.util_save import printx,save_parameters
from utils.util_gfro import (obtain_fragment,
                       rotated_hamiltonian,
                       boson_eigenspectrum_sparse,
                       boson_eigenspectrum_full,
                       quad_diagonalization)
from Henon_Heiles.critical_energy import critical_energy

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# number of modes, for HH Hamiltonian, we have x, y two modes
n = 2

# Occupation number (Number of basis functions)

# Henon-Heiles system Hamiltonian
lam = 0.09
h_variables = [1/2, 1/2, lam, - lam / 3]
H = henon_heiles(h_variables)

truncations = [20, 30, 40]
energy_changes = []
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
    ranges = 200

    energies = []

    # Loop over all the eigenstates
    for i in range(ranges):
        ee2 = e1[:, i]  # Extract i-th eigenstate
        ff1 = ee2.reshape(truncation, truncation)
        energies.append(round(eigenvalues1[i], 7))

    energy_changes.append(np.array(energies))

differences = []
for i in range(1, len(energy_changes)):
    differences.append(abs(energy_changes[i] - energy_changes[i - 1]))

print(differences)


