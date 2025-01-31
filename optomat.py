import numpy as np

## functions to build each matrix elements for b, b^dagger, q, q^2, q^3, q^4, p, and p^2
# Harmonic oscillator basis

def b1(i, j): # Define each matrix elements of b
    b1 = 0
    
    if i == j - 1:
        b1 = ((i+1)**0.5) 
        
    return b1

def b2(i, j): # Define each matrix elements of b^2
    b2 = 0
    
    if i == j - 2:
        b2 = (((i+2)*(i+1))**0.5) 
        
    return b2

def b3(i, j): # Define each matrix elements of b^3
    b3 = 0
    
    if i == j - 3:
        b3 = (((i+3)*(i+2)*(i+1))**0.5) 
        
    return b3

def b4(i, j): # Define each matrix elements of b^4
    b4 = 0
    
    if i == j - 4:
        b4 = (((i+4)*(i+3)*(i+2)*(i+1))**0.5) 
        
    return b4

def bd1(i, j): # Define each matrix elements of b^dagger
    bd1 = 0
    
    if i == j + 1:
        bd1 = ((j+1)**0.5) 
        
    return bd1

def bd2(i, j): # Define each matrix elements of (b^dagger)^2
    bd2 = 0
    
    if i == j + 2:
        bd2 = (((j + 2) * (j + 1))**0.5) 
        
    return bd2

def bd3(i, j): # Define each matrix elements of (b^dagger)^3
    bd3 = 0
    
    if i == j + 3:
        bd3 = (((j + 3) *(j + 2) * (j + 1))**0.5) 
        
    return bd3

def bd4(i, j): # Define each matrix elements of (b^dagger)^4
    bd4 = 0
    
    if i == j + 4:
        bd4 = (((j + 4) *(j + 3) *(j + 2) * (j + 1))**0.5) 
        
    return bd4

def bdb(i, j):# Define each matrix elements of b^dagger b
    x2 = 0
    
    if i == j:
        x2 = i
                           
    return x2

def bbd(i, j):# Define each matrix elements of  bb^dagger
    x2 = 0
    
    if i == j:
        x2 = i+1
                           
    return x2

def x1(i, j): # Define each matrix elements of q
    x1 = 0
    
    if i == j - 1:
        x1 = ((i+1)**0.5) / (2**0.5)
        
    if i == j + 1:
        x1 = ((j+1)**0.5) / (2**0.5)
        
    return x1

def x2(i, j):# Define each matrix elements of q^2
    x2 = 0
    
    if i == j:
        x2 = (2 * (j) + 1) / 2
        
    if i == j + 2:
        x2 = (((j + 2) * (j + 1)) ** 0.5) / 2
        
    if i == j - 2:
        x2 = (((i + 1) * (i + 2)) ** 0.5) / 2
                           
    return x2

def x3(i, j): # Define each matrix elements of q^3
    x3 = 0
    
    if i == j + 1:
        x3 = (3 * ((j+1) ** 1.5)) / (2 * (2 ** 0.5))
        
    if i == j - 1:
        x3 = (3 * ((j) ** 1.5)) / (2 * (2 ** 0.5))
        
    if i == (j + 3):
        x3 = (((j + 2) * (j + 2) * (j + 1)) ** 0.5) / (2 * (2 ** 0.5))
        
    if i == (j - 3):
        x3 = (((i + 1) * (i + 2) * (i + 3)) ** 0.5) / (2 * (2 ** 0.5))
        
    return x3

def x4(i, j): # Define each matrix elements of q^4
    x4 = 0
    
    if i == j:
        x4 = (6 * (j) ** 2 + 6 * (j) + 3) / 4
        
    if i == j + 2:
        x4 = ((4 * (j) + 6) * (((j + 1) * (j + 2)) ** 0.5)) / 4
        
    if i == (j - 2):
        x4 = ((4 * (i + 2) - 2) * (((i + 1) * (i + 2)) ** 0.5)) / 4
        
    if i == (j - 4):
        x4 = (((i + 1) * (i + 2) * (i + 3) * (i + 4)) ** 0.5) / 4
        
    if i == (j + 4):
        x4 = (((j + 1) * (j + 2) * (j + 3) * (j + 4)) ** 0.5) / 4
           
    return x4

def xd1(i, j): # Define each matrix elements of p
    # the complex i is not multiplied here (careful!!)
    xd1 = 0
    
    if i == j - 1:
        xd1 = (-(i+1)**0.5) / (2**0.5)
        
    if i == j + 1:
        xd1 = ((j+1)**0.5) / (2**0.5)
        
    return xd1

def xd2(i, j): # Define each matrix elements of p^2
    
    # -ve sign is included here
    
    xd2 = 0
    
    if i == j:
        xd2 = (2 * (j) + 1) / 2
        
    if i == j + 2:
        xd2 = -(((j + 1) * (j + 2)) ** 0.5) / 2
        
    if i == j - 2:
        xd2 = -(((i + 1) * (i + 2)) ** 0.5) / 2
                           
    return xd2


## functions to build matrix representations for q, q^2, q^3, q^4, p and p^2

# q = 1/sqrt(2) * (b + b^dagger)
# p = -1/sqrt(2) * (b - b^dagger)  # there should be an i (complex) but not included here
# q^2 = 1/2 * (b^2 + b b^dagger + b^dagger b + b^dagger^2)
# p^2 = -1/2 * (b^2 - b b^dagger - b^dagger b + b^dagger^2)
# p^2 + q^2 = 2 * b^dagger b + 1
# H_0 = 1/2 * Omega (p^2 + q^2) = Omega (b^dagger b + 1/2)

def bm1(n):
    # Define matrix representation of b   
    b11 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            b11[i1, i2] = b1(i1, i2)  
    return b11

def bm2(n):
    # Define matrix representation of b^2   
    b22 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            b22[i1, i2] = b2(i1, i2)  
    return b22

def bm3(n):
    # Define matrix representation of b^3   
    b33 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            b33[i1, i2] = b3(i1, i2)  
    return b33

def bm4(n):
    # Define matrix representation of b^4   
    b44 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            b44[i1, i2] = b4(i1, i2)  
    return b44

def bdm1(n):
    # Define matrix representation of b^dagger   
    bd11 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            bd11[i1, i2] = bd1(i1, i2)  
    return bd11

def bdm2(n):
    # Define matrix representation of (b^dagger)^2   
    bd22 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            bd22[i1, i2] = bd2(i1, i2)  
    return bd22

def bdm3(n):
    # Define matrix representation of (b^dagger)^3  
    bd33 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            bd33[i1, i2] = bd3(i1, i2)  
    return bd33

def bdm4(n):
    # Define matrix representation of (b^dagger)^2   
    bd44 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            bd44[i1, i2] = bd4(i1, i2)  
    return bd44

def bdbm2(n):
    # Define matrix representation of b^dagger b   
    b11 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            b11[i1, i2] = bdb(i1, i2)  
    return b11

def bbdm2(n):
    # Define matrix representation of b b^dagger    
    b11 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            b11[i1, i2] = bbd(i1, i2)  
    return b11

def xx1(n):
    # Define matrix representation of q   
    x11 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            x11[i1, i2] = x1(i1, i2)  
    return x11

def xx2(n):
    # Define matrix representation of q^2
    x22 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            x22[i1, i2] = x2(i1, i2)  
    return x22

def xx3(n):
    # Define matrix representation of q^3
    x33 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            x33[i1, i2] = x3(i1, i2) 
    return x33

def xx4(n):
    # Define matrix representation of q^4
    x44 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            x44[i1, i2] = x4(i1, i2)  
    return x44

def xxd1(n):
    # Define matrix representation of p
    # the complex i is not multiplied here too (careful!!)
    xd11 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            xd11[i1, i2] = xd1(i1, i2)  
    return xd11

def xxd2(n):
    # Define matrix representation of p^2
    xd22 = np.zeros((n, n))
    for i1 in range(0,n):
        for i2 in range(0,n):
            xd22[i1, i2] = xd2(i1, i2)  
    return xd22