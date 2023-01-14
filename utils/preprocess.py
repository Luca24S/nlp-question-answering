import numpy as np
from pathlib import Path

def GloVe_extractData(pathGloveFile):
    """
    Extracts GloVe embeddings from a file and returns a dictionary of vectors.

    Args:
        pathGloveFile (Path): Path to the GloVe embeddings file.

    Returns:
        tuple: (gloveDict, gloveVal)
            - gloveDict (dict): Word-to-vector mapping.
            - gloveVal (np.ndarray): Array of all GloVe vectors.
    """
    gloveDict = {}
    with open(pathGloveFile, "r", encoding="utf8") as g:
        for line in g:
            word, *vector = line.split()
            gloveDict[word] = np.array(vector, dtype=np.float32)
    gloveVal = np.vstack(gloveDict.values())
    return gloveDict, gloveVal

def UnknownWord_careFnc(gloveVectors, unknValue, gloveDict):
    """
    Handles unknown words by assigning them random vectors.

    Args:
        gloveVectors (np.ndarray): GloVe vector space.
        unknValue (str): The unknown word.
        gloveDict (dict): Word-to-vector mapping.

    Returns:
        np.ndarray: Vector for the unknown word.
    """
    variance = np.var(gloveVectors, axis=0)
    mean = np.mean(gloveVectors, axis=0)
    random_vector = np.random.RandomState().multivariate_normal(mean, np.diag(variance))
    gloveDict[unknValue] = random_vector
    return random_vector

def seq2seq(lineSentence, gloveDict, gloveVectors):
    """
    Converts a sentence into a sequence of vectors using GloVe embeddings.

    Args:
        lineSentence (str): Sentence to tokenize and vectorize.
        gloveDict (dict): GloVe dictionary.
        gloveVectors (np.ndarray): GloVe vector space.

    Returns:
        tuple: (np.ndarray, list)
            - Sentence vectors.
            - Tokenized words.
    """
    tokens = lineSentence.strip('"(),-').lower().split()
    vectors, words = [], []
    for token in tokens:
        if token in gloveDict:
            vectors.append(gloveDict[token])
        else:
            vectors.append(UnknownWord_careFnc(gloveVectors, token, gloveDict))
        words.append(token)
    return np.array(vectors), words

def find_context(setFile, gloveDict, gloveVectors):
    """
    Processes the dataset into context, question, and answer tuples.

    Args:
        setFile (Path): Path to the dataset file.
        gloveDict (dict): GloVe dictionary.
        gloveVectors (np.ndarray): GloVe vector space.

    Returns:
        list: Processed dataset.
    """
    dataList, contextList = [], []
    with open(setFile, "r", encoding="utf8") as file:
        for line in file:
            num, content = line.split(" ", 1)
            if num == '1':
                contextList = []
            if "\t" in content:
                question, answer, _ = content.split("\t")
                dataList.append((tuple(zip(*contextList)) + seq2seq(question, gloveDict, gloveVectors) + seq2seq(answer, gloveDict, gloveVectors)))
            else:
                contextList.append(seq2seq(content.strip(), gloveDict, gloveVectors))
    return dataList