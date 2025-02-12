import numpy as np
import scipy.linalg

from Henon_Heiles.critical_energy import unbound_criterion
from utils.util_hamil import henon_heiles, test,test1,test2,test5, H_real
from openfermion import get_sparse_operator, QuadOperator, get_boson_operator, normal_ordered
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
truncation = 40

sparse_op = get_sparse_operator(H, trunc=truncation)

full_matrix = sparse_op.toarray()

single_site_operator = H_real(normal_ordered(get_boson_operator(QuadOperator('q0 q0', 1))))
single_site_op_sparse = get_sparse_operator(single_site_operator, trunc=truncation).toarray()
full_space_single_op = np.kron(np.eye(truncation), single_site_op_sparse)


def get_time_evolution(t):
    unitary_matrix = scipy.linalg.expm(-1j * full_matrix * t)
    time_evolved = unitary_matrix.conj().T @ full_space_single_op @ unitary_matrix
    frobenius_norm = np.linalg.norm(time_evolved, 'fro')
    time_evolved /= frobenius_norm

    # state_representation = time_evolved.reshape(-1)
    #
    # ff1 = np.outer(state_representation, state_representation)

    # state_representation = time_evolved.reshape(-1)

    entropy = von_neumann_entropy_rdm(time_evolved, truncation)

    return entropy

t_values = np.linspace(0, 10, 100)  # Adjust range as needed

entropy_values = [get_time_evolution(t) for t in t_values]

# Plot entropy vs. time
plt.figure(figsize=(8, 5))
plt.plot(t_values, entropy_values, label="Entropy", color="blue")
plt.xlabel("Time (t)")
plt.ylabel("Von Neumann Entropy")
plt.title("Time Evolution of Entropy")
plt.legend()
plt.grid(True)
plt.show()
