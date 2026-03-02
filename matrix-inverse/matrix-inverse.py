import numpy as np
from numpy.linalg import inv, det

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A, dtype=float)
    
    # Check if A is a 2D square matrix
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        # Check if the matrix is not singular (determinant is not zero)
        if det(A) != 0:
            return inv(A)
            
    return None