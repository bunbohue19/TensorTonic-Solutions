import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = X.shape[0]

    # Init learnable params
    w, b = np.zeros((X.shape[1], 1)), 0.0

    for i in range(steps):
        # Compute probability
        p = _sigmoid(np.dot(X, w) + b)
        # Compute gradients
        loss = p - y.reshape(-1, 1)
        w_grad = np.dot(X.T, loss) / N
        b_grad = np.sum(loss) / N
        # Update params
        w = w - lr * w_grad
        b = b - lr * b_grad

    # Standardize
    w = w.reshape(-1)
    b = float(b)
    
    return (w, b)