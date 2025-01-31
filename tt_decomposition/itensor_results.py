import numpy as np
from itensor import Index, MPS, randomMPS

# Define the dimensions of the tensor (3x3x3 for simplicity)
dim1, dim2, dim3 = 3, 3, 3

# Create a simple tensor with shape (3, 3, 3)
tensor = np.arange(1, dim1 * dim2 * dim3 + 1).reshape((dim1, dim2, dim3))
print("Original Tensor:\n", tensor)

# Step 1: Define the indices for the MPS (each corresponding to a mode of the tensor)
i = Index(dim1, "i")
j = Index(dim2, "j")
k = Index(dim3, "k")

# Step 2: Convert the numpy tensor to an ITensor
from itensor import ITensor
A = ITensor(i, j, k)

# Assign values from the numpy array to the ITensor
for i1 in range(1, dim1 + 1):
    for j1 in range(1, dim2 + 1):
        for k1 in range(1, dim3 + 1):
            A[i1, j1, k1] = tensor[i1 - 1, j1 - 1, k1 - 1]

# Step 3: Convert the ITensor to an MPS
# Randomly initialize an MPS with 3 sites, matching the dimensions of the tensor
mps = randomMPS([i, j, k])

# Perform decomposition using the ITensor's SVD functionality
from itensor import svd
left, singular_values, right = svd(A, i)  # Decompose along the first dimension

# Print the left, singular values, and right tensor shapes
print(f"\nLeft Tensor (Core 1): Shape = {left.shape()}")
print(f"Singular Values Tensor: Shape = {singular_values.shape()}")
print(f"Right Tensor (Core 2 and Core 3 combined): Shape = {right.shape()}")
