import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tensor_train # matrix_product_state
from tensorly import tt_to_tensor

## MPS rank 1
def mps1(f1):
    mps_factors = tensor_train(f1, rank=[1,1,1,1])
    mps_differ = f1-tt_to_tensor(mps_factors)
    
    return np.round(np.sum(np.abs(mps_differ)),7)


## MPS rank 2
def mps2(f1):
    mps_factors = tensor_train(f1, rank=[1,2,2,1])
    mps_differ = f1-tt_to_tensor(mps_factors)

    return np.round(np.sum(np.abs(mps_differ)),7)


## CP rank 1
def cpd1(f1):
    cp_factors = parafac(f1, rank=1)
    cp_differ= f1-tl.cp_to_tensor(cp_factors)
    
    return np.round(np.sum(np.abs(cp_differ)),7)

## CP rank 2
def cpd2(f1):
    cp_factors = parafac(f1, rank=2)
    cp_differ= f1-tl.cp_to_tensor(cp_factors)
    
    return np.round(np.sum(np.abs(cp_differ)),7)
