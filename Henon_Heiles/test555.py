import time
import csv
import inspect
import numpy as np
from openfermion import get_sparse_operator
from optomat import xx1,xx2,xx3,xxd2
from util_hamil import test0,test,test1,test2,test5
from util_mutualinfo import mutual_information
from util_covar import covariance,covariance_x2
from ten_network import mps1,cpd1,mps2,cpd2
from util_save import printx,save_parameters
from util_gfro import (obtain_fragment,
                       rotated_hamiltonian, 
                       boson_eigenspectrum_sparse,
                       boson_eigenspectrum_full,
                       quad_diagonalization)
import matplotlib.pyplot as plt

# Define variables
n = 2  # number of modes

lam = 0.090
# Define Hamiltonian using parameters
h_variables = [1/2, 1/2, lam*1, lam*(-1/3)]  # variables go into the Hamiltonian

# Initialize lists to store truncation values and corresponding eigenvalues
trunc_values = range(4, 15)  # Adjust range as needed (e.g., from 3 to 10)
eigenvalue_convergence = []

# Loop over different truncation values
for truncation in trunc_values:
    H = test0(h_variables)  # Generate Hamiltonian using OpenFermion Bosonic Operators

    # Get the full eigenspectrum for the current truncation value
    sparse_op = get_sparse_operator(H, trunc=truncation)
    Hf = sparse_op.toarray()
    eigenvalues, _ = np.linalg.eigh(Hf)
    
    # Store the lowest eigenvalue (or adjust to store multiple if needed)
    eigenvalue_convergence.append(eigenvalues[:15])  # Storing the first 3 eigenvalues for each truncation

# Convert to numpy array for easy plotting
eigenvalue_convergence = np.array(eigenvalue_convergence)

# Plot eigenvalues against truncation values
plt.figure(figsize=(8, 6))
for i in range(eigenvalue_convergence.shape[1]):
    plt.plot(trunc_values, eigenvalue_convergence[:, i], label=f'Eigenvalue {i}')
    
plt.xlabel('Truncation Value')
plt.ylabel('Eigenvalues')
plt.title('Convergence of Eigenvalues with Truncation')
#plt.legend()
plt.grid(True)
plt.show()