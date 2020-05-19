"""
    A program to GloVe vectors to text analysis
    This program test 2 files to get a feel for analyzing text.
    It seems that means are not a great summary of the documents in this space.
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
            # make all the works lower case.
            items = list(map(str.lower, items))
            # remove punctuation from the end of words
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


def reduce_mean_word_list(words, word_map, matrix):
    """
    Take a list of words and summarize them as a vector using 'mean'.
    returns a numpy vector
    :param words:
    :param word_map:
    :param matrix:
    :return:
    """
    vectors = []
    for word in words:
        if word in word_map:
            index = word_map[word]
            vectors.append(matrix[index])
    vectors = np.array(vectors)
    mean_vec = np.mean(vectors, axis=0)
    return mean_vec


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
        if word in word_map:
            index = word_map[word]
            vec = vec + matrix[index]
    return vec


def main():
    embedding_filename = '../data/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)
    print('embeddings loaded')

    # create some test words
    love_vec = matrix[word_map['love']]
    evil_vec = matrix[word_map['evil']]
    similarity = cossim(evil_vec, love_vec)
    print("cosine sim('evil', 'love')=", similarity)

    print('+++++ MEAN +++++')

    # calculate text 1 words and vector
    test1_words = load_text_words('../data/test1.txt')
    mean_vec1 = reduce_mean_word_list(test1_words, word_map, matrix)

    # calculate text 2 words and vector
    test2_words = load_text_words('../data/test2.txt')
    mean_vec2 = reduce_mean_word_list(test2_words, word_map, matrix)

    # calculate similarity
    similarity = cossim(mean_vec1, mean_vec2)
    print("cosine sim(text 1 mean, test 2 mean)", similarity)

    similarity = cossim(mean_vec1, love_vec)
    print("cosine sim(text 1 mean, 'love')=", similarity)
    similarity = cossim(mean_vec1, evil_vec)
    print("cosine sim(text 1 mean, 'evil')=", similarity)

    similarity = cossim(mean_vec2, love_vec)
    print("cosine sim(text 2 mean, 'love')=", similarity)
    similarity = cossim(mean_vec2, evil_vec)
    print("cosine sim(text 2 mean, 'evil')=", similarity)

    print("Mean does not seem to work well.")

    print('+++++ SUM +++++')
    # calculate text 1 words and vector
    test1_words = load_text_words('../data/test1.txt')
    sum_vec1 = reduce_sum_word_list(test1_words, word_map, matrix)

    # calculate text 2 words and vector
    test2_words = load_text_words('../data/test2.txt')
    sum_vec2 = reduce_sum_word_list(test2_words, word_map, matrix)


    # calculate similarity
    similarity = cossim(sum_vec1, sum_vec2)
    print("cosine sim(text 1 sum, test 2 sum)", similarity)

    similarity = cossim(sum_vec1, love_vec)
    print("cosine sim(text 1 sum, 'love')=", similarity)
    similarity = cossim(sum_vec1, evil_vec)
    print("cosine sim(text 1 sum, 'evil')=", similarity)

    similarity = cossim(sum_vec2, love_vec)
    print("cosine sim(text 2 sum, 'love')=", similarity)
    similarity = cossim(sum_vec2, evil_vec)
    print("cosine sim(text 2 sum, 'evil')=", similarity)

    print('Sum is equivalent to mean.')
    print('Mean is a type of scaling.')
    print('cosine similarity is invariance to scaling.')

    print('+++++++++ Differnt Approach +++++++++')

    feeling_vec = matrix[word_map['feeling']]
    similarity = cossim(evil_vec-feeling_vec, love_vec-feeling_vec)
    print("cosine sim('evil' - 'feeling', 'love' - 'feeling')=", similarity)

    query = 'character'
    feeling_vec = matrix[word_map[query]]
    similarity = cossim(evil_vec - feeling_vec, love_vec - feeling_vec)
    print("cosine sim('evil' - '%s', 'love' - '%s')=" % (query, query), similarity)


main()
