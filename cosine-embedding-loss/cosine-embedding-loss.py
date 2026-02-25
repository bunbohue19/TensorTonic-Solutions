import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1, ord=2) * np.linalg.norm(x2, ord=2))
    return float(np.where(label == 1, 1 - cos_sim, np.maximum(0, cos_sim - margin)))