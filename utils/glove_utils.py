import numpy as np

def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings from a file.

    Args:
        glove_path (str or Path): Path to the GloVe file.

    Returns:
        dict: A dictionary where keys are words and values are embedding vectors.
    """
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = vector
    return glove_dict

def get_glove_vector(word, glove_dict, default_vector=None):
    """
    Retrieve the GloVe vector for a given word.

    Args:
        word (str): The word to look up.
        glove_dict (dict): Dictionary of GloVe vectors.
        default_vector (np.ndarray): Vector to return if the word is not found.

    Returns:
        np.ndarray: GloVe vector for the word.
    """
    if word in glove_dict:
        return glove_dict[word]
    elif default_vector is not None:
        return default_vector
    else:
        raise ValueError(f"Word '{word}' not found and no default vector provided.")

def create_random_vector(glove_vectors, seed=None):
    """
    Create a random vector based on the distribution of GloVe embeddings.

    Args:
        glove_vectors (np.ndarray): Array of GloVe vectors.
        seed (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: A random vector.
    """
    np.random.seed(seed)
    mean = np.mean(glove_vectors, axis=0)
    std = np.std(glove_vectors, axis=0)
    return np.random.normal(mean, std)