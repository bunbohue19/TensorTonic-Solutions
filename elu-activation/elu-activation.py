import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    return [np.where(sample > 0, sample, alpha * (np.exp(sample) - 1)) for sample in x]