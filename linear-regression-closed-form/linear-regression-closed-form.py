import numpy as np
from numpy.linalg import inv

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = inv(X.T @ X) @ X.T @ y
    return w