"""
    This program works to search embeddings for similar and dissimilar words.
    Use this for an interactive way to explore the embedding space.
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


def load_text_words(filename):
    """
    This function takes a text document and creates a list of works for the document.
    It returns the list of text words.
    :param filename:
    :return: text_words
    """
    text_words = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            items = line.split()
            items = list(map(str.lower, items))
            items = list(map(lambda x: x.strip('.,[]!?;:'), items))
            text_words.extend(items)
    # return a list of words from the text.
    return text_words


def cossim(vA, vB):
    """
    Calcuate the cosine similarity value.
    Returns the similarity value, range: [-1, 1]
    :param vA:
    :param vB:
    :return: similarity
    """
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))


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


def main():
    embedding_filename = '../data/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)

    while True:
        query_word = input('enter query word>')
        print('==============================')
        print('similar to ', query_word)
        search_similar(word_map, matrix, query_word, 0.70)

        print('------------------')
        print('dissimilar to ', query_word)
        search_dissimilar(word_map, matrix, query_word, -0.70)


main()
