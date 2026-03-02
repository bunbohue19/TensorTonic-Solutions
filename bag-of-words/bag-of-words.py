import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Create a mapping from word to index for O(1) lookups
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # Initialize the vector with zeros
    vector = np.zeros(len(vocab), dtype=int)
    
    # Count occurrences
    for token in tokens:
        if token in word_to_index:
            index = word_to_index[token]
            vector[index] += 1
            
    return vector