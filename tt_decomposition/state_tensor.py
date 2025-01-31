import numpy as np


def get_coeff_tensor(site_dim: int, n_sites: int, n_elec: int, spin: int):
    """
    Get the coefficient tensor of a randomly generated state.
    :param n_sites: Number of orbitals / spin-orbitals
    :return:
    """
    tensor_shape = (site_dim, ) * n_sites

    # 1-D representation of the coefficients
    coeff_1d = np.random.rand(site_dim ** n_sites)

    for i in range(site_dim ** n_sites):
        base_4 = to_base_4(i)
        if base_4_to_num_elec(base_4) != n_elec:
            coeff_1d[i] = 0

        if get_spin(base_4) != spin:
            coeff_1d[i] = 0


    # Select ones that satisfy the number of electrons configuration


    # Rank-n_sites tensor representation of the coefficients
    coefficient_tensor = np.reshape(coeff_1d, tensor_shape)
    return coefficient_tensor


def to_base_4(number):
    if number == 0:
        return "0"
    base_4_digits = []
    while number > 0:
        base_4_digits.append(str(number % 4))  # Get the remainder when divided by 4
        number //= 4  # Update number to the quotient for the next iteration
    base_4_digits.reverse()  # Reverse the list to get the correct order
    return ','.join(base_4_digits)


def base_4_to_num_elec(base_4_digits):
    digit_list = base_4_digits.split(",")
    sum_elec = 0
    for digit in digit_list:
        if digit == "1" or digit == "2":
            sum_elec += 1
        elif digit == "3":
            sum_elec += 2

    return sum_elec


def get_spin(base_4_digits):
    digit_list = base_4_digits.split(",")
    sum_spin = 0
    for digit in digit_list:
        if digit == "1":
            sum_spin += 1
        elif digit == "2":
            sum_spin -= 1

    return sum_spin


if __name__ == '__main__':
    base_4 = to_base_4(10)
    print(base_4)
    print(base_4_to_num_elec(base_4))
    n_elec = 4
    site_dim = 4
    n_sites = 3
    spin = 0
    coefficient_tensor = get_coeff_tensor(site_dim, n_sites, n_elec, spin)
    coeff_1d = np.reshape(coefficient_tensor, (1, site_dim ** n_sites))
    tensor_shape = coefficient_tensor.shape
    tensor_n = tensor_shape[0]
    tensor_rank = len(tensor_shape)
    print(coefficient_tensor)
    print(coeff_1d)
