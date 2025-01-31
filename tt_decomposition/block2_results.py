import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from state_tensor import get_coeff_tensor
from pyscf import gto, scf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg

# Create a sample 3D tensor (shape = 4x4x4)
# tensor = np.random.random((4, 4, 4))

# print("Original Tensor Shape:", tensor.shape)


def coeff_to_MPS(coeff_tensor):
    tensor_shape = coeff_tensor.shape
    tensor_n = tensor_shape[0]
    tensor_rank = len(tensor_shape)
    n_dets = tensor_n ** tensor_rank
    coeff_1d = np.reshape(coeff_tensor, (n_dets, ))

    if tensor_n != 4:
        raise ValueError("Each site must have 4 dimension")

    csf_matrix = []
    compact_coeff_1d = []
    sz_list = [0, 1, 2, 3]
    non_zero_count = 1
    for k in range(n_dets):
        if coeff_1d[k] != 0:
            det_list = [0, 0, 0, 0]
            count = 0
            number = k
            while number > 0:
                remainder = number % 4
                det_list[count] = sz_list[remainder]
                number //= 4
                count += 1
            csf_matrix.append(det_list)
            compact_coeff_1d.append(coeff_1d[k])

    dvals = compact_coeff_1d

    print(csf_matrix)
    spin = 0
    print(len(dvals))
    print(len(csf_matrix))

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ,
                        n_threads=4)
    driver.initialize_system(n_sites, n_elec, spin, orb_sym=None)

    MPS = driver.get_mps_from_csf_coefficients(csf_matrix, dvals, "CMPS", 2)
    return MPS


def generate_csf_matrix(n_dets, n_sites, mode='SZ'):
    """
    Generate an occupancy matrix for CSFs/DETs along with associated coefficients.

    Args:
    - n_dets: Number of configurations (determinants).
    - n_sites: Number of sites (orbitals).
    - mode: Either 'SU2' or 'SZ'.

    Returns:
    - csf_matrix: Array of shape (n_dets, n_sites) representing occupancies.
    - coefficients: Array of shape (n_dets,) representing coefficients for each CSF.
    """
    if mode not in ['SU2', 'SZ']:
        raise ValueError("Mode must be either 'SU2' or 'SZ'.")

    # Randomly generate CSF matrix with specified occupancy values
    occupancy_values = {
        'SU2': [0, 1, 2, 3],
        # 0: empty, 1: spin-up, 2: doubly occupied, 3: spin-down
        'SZ': [0, 1, 2, 3]  # 0: empty, 1: alpha, 2: doubly occupied, 3: beta
    }

    # Initialize CSF matrix
    csf_matrix = np.zeros((n_dets, n_sites), dtype=int)

    # Randomly fill the occupancy matrix
    for i in range(n_dets):
        for j in range(n_sites):
            csf_matrix[i, j] = np.random.choice(occupancy_values[mode])

    # Generate random coefficients for each CSF (can be adjusted as needed)
    coefficients = np.random.rand(n_dets)

    # Normalize coefficients
    coefficients /= np.linalg.norm(coefficients)

    return csf_matrix, coefficients


if __name__ == "__main__":
    site_dim = 4
    n_sites = 4
    n_elec = 4
    spin = 0
    coefficient_tensor = get_coeff_tensor(site_dim, n_sites, n_elec, spin)

    MPS = coeff_to_MPS(coefficient_tensor)

    print(MPS)
    # Assuming 'tensor' is your original tensor representing CI states
    # For example, a 4x4x4x4 tensor (as in your earlier question)
    tensor = np.random.rand(4, 4, 4,
                            4)  # This should be replaced with your actual tensor

    # Step 1: Reshape the tensor to a 2D array if necessary
    # This assumes you want to flatten the tensor into a matrix of coefficients
    # The shape of the new matrix will depend on your specific application
    # Here we reshape to (num_configs, num_coeffs)

    # Flatten the tensor into a 2D matrix where rows represent configurations
    num_configs = tensor.shape[0] * tensor.shape[
        1]  # Example configuration count
    num_coeffs = tensor.shape[2] * tensor.shape[3]  # Example coefficient count

    # Example parameters
    n_dets = 4 ** 4  # Number of determinants
    n_sites = 4  # Number of sites

    # Generate CSF matrix and coefficients in SU2 mode
    csf_matrix, coefficients = generate_csf_matrix(n_dets, n_sites, mode='SZ')

    # Print results
    # print("CSF/DET Matrix:\n", csf_matrix)
    # print("Associated Coefficients:\n", coefficients)

    n_sites = n_sites
    n_elec = n_sites
    spin = 0
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ,
                        n_threads=4)
    driver.initialize_system(n_sites, n_elec, spin)

    MPS = driver.get_mps_from_csf_coefficients(csf_matrix, coefficients, "mps2", 2)
    # print("MPS Matrix:\n", MPS)

