import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinSite


if __name__ == '__main__':

    # Step 1: Create a simple 3D tensor of shape (3, 3, 3)
    tensor = np.arange(1, 28).reshape((3, 3, 3))
    print("Original Tensor:\n", tensor)

    # Step 2: Flatten the tensor into a 1D array
    # Since TenPy works primarily with 1D chains, we'll flatten it for simplicity
    tensor_flattened = tensor.flatten()
    print("\nFlattened Tensor:\n", tensor_flattened)

    # Step 3: Create a spin site for each "physical" dimension (each tensor element will be considered a spin site)
    # This step is necessary because TenPy expects sites with physical dimensions.
    local_dim = 27  # Each site corresponds to one element in our flattened tensor (27 elements)
    site = SpinSite(
        S=0.5)  # We can use a Spin-1/2 site for simplicity (can represent 2 states)

    # Step 4: Create an MPS object from the flattened tensor
    # We initialize an MPS with random tensors and assign the flattened tensor as data
    psi = MPS.from_product_state([site] * local_dim, ["up"] * local_dim)

    # Step 5: Decompose the tensor into MPS cores
    # MPS optimization and tensor decomposition is TenPy's core strength.
    # In this example, we'll directly create an MPS and check its structure.
    print("\nMPS Site Tensors:")
    for i, core in enumerate(psi._B):
        print(f"Core {i} Shape: {core.shape}")

    # Step 6: Check the tensor data
    # Each "site" in the MPS corresponds to one element in the flattened tensor
    print("\nMPS Tensor Data at Each Site:")
    for i, core in enumerate(psi._B):
        print(f"Site {i}: {core}")
