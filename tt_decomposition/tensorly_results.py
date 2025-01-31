import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor


def get_mps(tensor, rank):
    tt_cores = tensor_train(tensor, rank=rank)
    # Display the cores in the TT decomposition (MPS format)
    print("\nTT Cores:")
    for i, core in enumerate(tt_cores):
        print(f"Core {i}: Shape = {core.shape}")

    # Convert TT cores back to the full tensor
    reconstructed_tensor = tt_to_tensor(tt_cores)

    # Check the reconstruction error
    error = tl.norm(tensor - reconstructed_tensor)
    print(f"\nReconstruction Error: {error}")

    return error, tt_cores


if __name__ == "__main__":

    # Create a sample 3D tensor (shape = 4x4x4)
    # tensor = np.random.random((4, 4, 4))

    # Step 1: Create a Hermitian rank-4 tensor (4x4x4x4 tensor for 4 sites)
    # Initialize a random 4x4 matrix
    base_tensor = np.array([[0.5, 0.1, 0.3, 0.2],
                            [0.1, 0.6, 0.2, 0.3],
                            [0.3, 0.2, 0.4, 0.5],
                            [0.2, 0.3, 0.5, 0.7]])

    # Create a 4x4x4x4 tensor using the base matrix
    tensor = np.zeros((4, 4, 4, 4))

    for i in range(4):
        for j in range(4):
            tensor[
                i, j] = base_tensor + base_tensor.T  # Symmetric part to ensure Hermitian property

    # Now make the tensor Hermitian
    tensor = (tensor + np.swapaxes(tensor, 2,
                                   3)) / 2  # Ensures V_{ijkl} = V_{klij}

    print("Original Tensor:\n", tensor)
    # print("Original Tensor Shape:", tensor.shape)

    # Perform Tensor Train (TT) decomposition
    tt_cores = tensor_train(tensor, rank=[1, 2, 2, 2, 1])  # Rank configuration for TT decomposition

    # Display the cores in the TT decomposition (MPS format)
    print("\nTT Cores:")
    for i, core in enumerate(tt_cores):
        print(f"Core {i}: Shape = {core.shape}")

    # Convert TT cores back to the full tensor
    reconstructed_tensor = tt_to_tensor(tt_cores)

    # Check the reconstruction error
    error = tl.norm(tensor - reconstructed_tensor)
    print(f"\nReconstruction Error: {error}")
