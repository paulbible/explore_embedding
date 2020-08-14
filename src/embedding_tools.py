"""
A tools file to collect some reusable embedding functions
"""
import numpy as np


def load_embeddings(filename):
    """
    This function loads the embedding from a file and returns 2 things
    1) a word_map, this is a dictionary that maps words to an index.
    2) a matrix of row vectors for each work, index the work using the vector.

    :param filename:
    :return: word_map, matrix
    """
    count = 0
    matrix = []
    word_map = {}
    with open(filename, encoding="utf8") as f:
        # with open(filename) as f:
        for line in f:
            line = line.strip()
            items = line.split()
            word = items[0]
            rest = items[1:]
            # print("word:", word)
            word_map[word] = count
            count += 1

            rest = list(map(float, rest))
            matrix.append(rest)
    matrix = np.array(matrix)
    return word_map, matrix


def reduce_sum_word_list(words, word_map, matrix):
    """
    Take a list of words and summarize them as a vector using 'mean'.
    returns a numpy vector
    :param words:
    :param word_map:
    :param matrix:
    :return:
    """
    vec = np.zeros(matrix.shape[1])
    for word in words:
        word = word.lower()
        if word in word_map:
            index = word_map[word]
            vec = vec + matrix[index]
    return vec


def reduce_sum_word_list_weighted(words, word_map, matrix, word_weight_map):
    """
    Take a list of words and summarize them as a vector using 'mean'.
    returns a numpy vector
    :param words:
    :param word_map:
    :param matrix:
    :param word_weight_map:
    :return:
    """
    vec = np.zeros(matrix.shape[1])
    for word in words:
        word = word.lower()
        if word in word_map:
            index = word_map[word]
            if word in word_weight_map:
                vec = vec + matrix[index]*word_weight_map[word]
            else:
                vec = vec + matrix[index]
    return vec


def cossim(vA, vB):
    """
    Calcuate the cosine similarity value.
    Returns the similarity value, range: [-1, 1]
    :param vA:
    :param vB:
    :return: similarity
    """
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))


def calc_similar_values(word_map, matrix, query):
    """
    Claculate the similarity of every word vector with the query word vector
    :param word_map:
    :param matrix:
    :param query:
    :return:
    """
    if query not in word_map:
        print(query, "not found")
        return None

    values = []
    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        values.append(cossim(test_vector, vector))
    return values


def search_dissimilar(word_map, matrix, query, threshold=-0.90):
    """
    Search through the matrix and get words that are similar to the give word
    using the embedding vectors.
    :param word_map:
    :param matrix:
    :param query:
    :param threshold:
    :return:
    """
    if query not in word_map:
        print(query, "not found")
        return None

    words = ['' for i in range(len(word_map))]
    for word in word_map:
        words[word_map[word]] = word

    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        if cossim(test_vector, vector) <= threshold:
            print(words[i])


def search_similar(word_map, matrix, query, threshold=0.95):
    """
    Search through the matrix and get words that are similar to the give word
    using the embedding vectors.
    :param word_map:
    :param matrix:
    :param query:
    :param threshold:
    :return:
    """
    if query not in word_map:
        print(query, "not found")
        return None

    words = ['' for i in range(len(word_map))]
    for word in word_map:
        words[word_map[word]] = word

    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        if cossim(test_vector, vector) >= threshold:
            print(words[i])