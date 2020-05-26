"""
    The program explores the distribution of similarity values for different words.
"""
import numpy as np
import matplotlib.pyplot as plt


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

    values = []
    num_rows = matrix.shape[0]
    vector = matrix[word_map[query]]
    for i in range(num_rows):
        test_vector = matrix[i]
        values.append(cossim(test_vector, vector))
    return values


def main():
    embedding_filename = '../data/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)

    while True:
        query_word = input('enter query word>')
        values = calc_similar_values(word_map, matrix, query_word)
        if values:
            plt.hist(values)
            plt.show()
        else:
            print("word %s not found" % query_word)


main()
