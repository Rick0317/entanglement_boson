import time
import csv
import inspect
import numpy as np
from tt_decomposition.bd_estimation import get_estimate
from mode3_utils.util_hamil import test,test1,test2,test5
from mode3_utils.util_mutualinfo import mutual_all
from mode3_utils.util_covar import covariance
from ten_network import mps1,cpd1,mps2,cpd2
from mode3_utils.util_save import printx,save_parameters
from mode3_utils.util_gfro import (obtain_fragment,
                       rotated_hamiltonian,
                       boson_eigenspectrum_sparse,
                       boson_eigenspectrum_full,
                       quad_diagonalization)



n = 3 # number of modes

# Define Hamiltonian using parameters

# h_variables = [1,1,1,0.6,0.6,0.6] # variables goes in to Hamiltonian

lam = 0.09

h_variables = [0.5, 0.5, 0.5, 0.01, 0.01, 0.01]

truncation = 3  # Occuppation number (Number of basis function)

def extract_function_name(func):
    return func.__name__

hamil_name = extract_function_name(test1)

H = test1(h_variables) # Generate Hamiltonian iterms of OpenFermion Bosonic Operators from "util_hamil.py"
                        #test1 contains only upto quadratic terms
maxit = 1

options = {
        'maxiter' : maxit, # maximum iteration goes to the cost function
        'gtol': 1e-7,  # Set the tolerance for cost function change
 #       'xatol': 1e-7,
        'disp'    : False
    }


start_time = time.time()

## Mutual information and MPS and CP errors before rotation
#eigenvalues1,e1 = boson_eigenspectrum_sparse(H, truncation, 1)
eigenvalues1,e1 = boson_eigenspectrum_full(H, truncation)
e1 = e1[:,0]
f1=e1.reshape(truncation,truncation, truncation)
print(eigenvalues1[0])
print(f"BD: {get_estimate(f1, 3, 10, 0.0001)}")

I_12, I_23, I_13 = mutual_all(f1, truncation) # Find mutual information between modes before rotation
covar_XY,covar_YZ,covar_XZ = covariance(e1, truncation) #  Find covariance between modes before rotation


parameters = []  #saving parameters
# paramters before rotation
parameters.append([round(I_12, 7), round(I_23, 7), round(I_13, 7), mps1(f1), mps2(f1), cpd1(f1), cpd2(f1),round(covar_XY, 7),round(covar_YZ, 7),round(covar_XZ, 7)])


params=[round(I_12, 7), round(I_23, 7), round(I_13, 7), mps1(f1), mps2(f1), cpd1(f1), cpd2(f1),round(covar_XY, 7),round(covar_YZ, 7),round(covar_XZ, 7)]

iteration = 0

printx(params,iteration)

## Lists of Cost functions

#=============================================================================
cost_fns = [
      # lambda I_12, I_23, I_13: I_12 + I_13,
      # lambda I_12, I_23, I_13: I_12 + I_23,
      # lambda I_12, I_23, I_13: I_23 + I_13,
      lambda I_12, I_23, I_13: I_12 + 5 * I_13 + I_23,
      # lambda I_12, I_23, I_13: I_12 + 5 * I_13 **(1/2) + I_23,
      # lambda I_12, I_23, I_13: I_12 + 5 * I_13 **(2) + I_23,
      # lambda I_12, I_23, I_13: I_12 + 5 * I_13 + I_23,
    #   lambda I_12, I_23, I_13: I_12 + 2 * I_13 **(1/3) + I_23,
    # lambda I_12, I_23, I_13: I_12 + 3 * I_13 **(1/3) + I_23,
    # lambda I_12, I_23, I_13: I_12 + 4 * I_13 **(1/3) + I_23,
    # lambda I_12, I_23, I_13: I_12 + 5 * I_13 **(1/3) + I_23,
    # lambda I_12, I_23, I_13: I_12 + 6 * I_13 **(1/3) + I_23,
    # lambda I_12, I_23, I_13: I_12 + 7 * I_13 **(1/3) + I_23,
    # lambda I_12, I_23, I_13: I_12 + 8 * I_13 **(1/3) + I_23,

      # lambda I_12, I_23, I_13: I_12,
      # lambda I_12, I_23, I_13: I_23,
      # lambda I_12, I_23, I_13: 5 * I_13
]
#=============================================================================

# =============================================================================
# cost_fns = [
#        lambda I_12, I_23, I_13: I_12 + 40 * I_13 + I_23,
#        lambda I_12, I_23, I_13: I_12 + 40 * I_13 ** (1/2) + I_23,
#        lambda I_12, I_23, I_13:  I_12  + 40 * I_13 ** (1/3)  + I_23
#          # lambda I_12, I_23, I_13: I_12 + 0 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 1 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 2 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 3 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 4 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 5 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 6 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 7 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 8 * I_13 + I_23,
#          # lambda I_12, I_23, I_13: I_12 + 16 * I_13 + I_23
#         ]
# =============================================================================



## Bogoliubov rotations with each cost function in the list

for fn in cost_fns: # runs over the list of cost functions


    result = obtain_fragment(H, n, options, fn) # parameters for rotation

    Hr = rotated_hamiltonian(H, result.x, n) # Hamiltonian in the rotated frame

    Hr = quad_diagonalization(Hr,n) #diagonalize upto the quadratic terms in the Hamiltonian

 #   eigenvalues2, e2 = boson_eigenspectrum_sparse(Hr, truncation, 1) # calculated the ground state evals and evecs

    eigenvalues2,e2 = boson_eigenspectrum_full(Hr, truncation)

    e2 = e2[:,0]

    print(eigenvalues2[0])

    f2 = e2.reshape(truncation, truncation, truncation) # reshape evec

    I_12, I_23, I_13 = mutual_all(f2, truncation) # Find mutual information between modes after rotation

    covar_XY,covar_YZ,covar_XZ = covariance(e2,truncation)

    parameters.append([round(I_12, 7), round(I_23, 7), round(I_13, 7), mps1(f2), mps2(f2), cpd1(f2), cpd2(f2),round(covar_XY, 7),round(covar_YZ, 7),round(covar_XZ, 7)]) # paramters after rotation with each cost function

    params=[round(I_12, 7), round(I_23, 7), round(I_13, 7), mps1(f2), mps2(f2), cpd1(f2), cpd2(f2),round(covar_XY, 7),round(covar_YZ, 7),round(covar_XZ, 7)]

    printx(params,iteration)

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print("Completed")


save_parameters(cost_fns,h_variables,maxit,parameters,hamil_name) #save parameters in to a csv file


