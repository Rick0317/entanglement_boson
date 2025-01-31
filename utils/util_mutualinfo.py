# These functions are for 3d models

import numpy as np


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # remove small rigenvalues
    return -np.sum(eigenvalues * np.log(eigenvalues))


def mutual_information(f1, n):

    rho_1, rho_2 = one_mode_rdm(f1, n)

    s_1 = von_neumann_entropy(rho_1)
    s_2 = von_neumann_entropy(rho_2)

    return 1 / 2 * (s_1 + s_2)


def von_neumann_entropy_rdm(f1, n):
    rho_1 = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                rho_1[i, j] += f1[
                    i * n + k, j * n + k]  # Trace over B


    s_1 = von_neumann_entropy(rho_1)
    return s_1


# single-mode RDMs

def one_mode_rdm(f1, n):

    rho_1 = np.zeros((n, n), dtype=complex)
    rho_2 = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            rho_1[i, j] = np.sum(f1[i, :] * np.conj(f1[j, :]))
            rho_2[i, j] = np.sum(f1[:, i] * np.conj(f1[:, j]))

    return rho_1, rho_2


def mutual_info_cost(f1, n, fn):

    # I_12,I_23,I_13 = mutual_info_full(f1,n)

    return mutual_information(f1, n)


if __name__ == '__main__':
    # rho = np.array([[1, 0], [0, 0]])
    state = np.array([1 , 0, 0,0])
    #state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    state = state.reshape(2, 2)
    print(f"MI: {mutual_information(state, 2)}")
    print(state.T)
    # f1 = np.array([[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]])
    f1 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    rho1, rho2 = one_mode_rdm(f1, 2)
    s1 = von_neumann_entropy(rho1)
    s2 = von_neumann_entropy(rho2)
    print(s1)
    print(s2)
