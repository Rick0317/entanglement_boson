from tensorly.decomposition import tensor_train
from tensorly import tt_to_tensor
import numpy as np


def get_estimate(input_vector, n_sites, max_bd, thershold):
    """
    Estimate the bond dimension required to represent the input_vector
    (Coefficient tensor) in the MPS format within the error of threshold.
    :param input_vector: The input coefficient tensor
    :param n_states: The number of
    :param max_bd: Maximum bond dimension for estimation
    :param thershold: The threshold for the error in the bond dimension
    :return: The approximated bond dimension.
    If the error didn't go below threshold, max_bd is returned
    """
    for bd in range(1, max_bd + 1):

        rank_list = [1]
        for i in range(n_sites - 1):
            rank_list.append(2)
        rank_list.append(1)

        mps_factors = tensor_train(input_vector, rank=bd)
        mps_differ = input_vector - tt_to_tensor(mps_factors)

        diff_num = np.round(np.sum(np.abs(mps_differ)), 7)

        print(diff_num)

        if diff_num < thershold:
            return bd

    return max_bd
