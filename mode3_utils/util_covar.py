import numpy as np

def tensor_product(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

def x1(n_max):
    m=1
    hbar=1
    omega=1
    """Create the position operator matrix for a single mode."""
    x = np.zeros((n_max, n_max))
    prefactor = np.sqrt(hbar / (2 * m * omega))
    for i in range(n_max - 1):
        x[i, i + 1] = x[i + 1, i] = prefactor * np.sqrt(i + 1)
    return x


def covariance(e1,truncation):

    X_mat = tensor_product(x1(truncation),np.eye(truncation),np.eye(truncation))
    Y_mat = tensor_product(np.eye(truncation),x1(truncation),np.eye(truncation))
    Z_mat = tensor_product(np.eye(truncation),np.eye(truncation),x1(truncation))
    
    exp_val_X = np.einsum('i,ij,j', e1.conj(), X_mat, e1).real
    exp_val_Y = np.einsum('i,ij,j', e1.conj(), Y_mat, e1).real
    exp_val_Z = np.einsum('i,ij,j', e1.conj(), Z_mat, e1).real
    
    XY_mat = tensor_product(x1(truncation),x1(truncation),np.eye(truncation))
    YZ_mat = tensor_product(np.eye(truncation),x1(truncation),x1(truncation))
    XZ_mat = tensor_product(x1(truncation),np.eye(truncation),x1(truncation))
    
    exp_val_XY = np.einsum('i,ij,j', e1.conj(), XY_mat, e1).real
    exp_val_YZ = np.einsum('i,ij,j', e1.conj(), YZ_mat, e1).real
    exp_val_XZ = np.einsum('i,ij,j', e1.conj(), XZ_mat, e1).real
    
    covar_XY = exp_val_XY - exp_val_X * exp_val_Y
    covar_YZ = exp_val_YZ - exp_val_Y * exp_val_Z
    covar_XZ = exp_val_XZ - exp_val_X * exp_val_Z
    
    return covar_XY,covar_YZ,covar_XZ 