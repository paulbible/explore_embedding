"""
    A program to GloVe vectors to text analysis
"""
import numpy as np


def load_embeddings(filename):
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
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))


def main():
    embedding_filename = '../data/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)

    test1_words = set(load_text_words('../data/test1.txt'))
    test_vectors = []
    for word in test1_words:
        # print('test1', word)
        if word in word_map:
            index = word_map[word]
            test_vectors.append(matrix[index])

    test_vectors = np.array(test_vectors)
    # print(test_vectors)
    # print(test_vectors.shape)
    mean_vec1 = np.mean(test_vectors, axis=0)
    # print(mean_vec1)
    # print(mean_vec1.shape)

    test2_words = set(load_text_words('../data/test2.txt'))
    test2_vectors = []
    for word in test2_words:
        # print('test2', word)
        if word in word_map:
            index = word_map[word]
            test2_vectors.append(matrix[index])
    test2_vectors = np.array(test2_vectors)
    # print(test2_vectors)
    # print(test2_vectors.shape)
    mean_vec2 = np.mean(test2_vectors, axis=0)
    # print(mean_vec2)
    # print(mean_vec2.shape)

    similarity = cossim(mean_vec1, mean_vec2)
    print("cosine similarity(text 1 mean, test 2 mean)", similarity)

    love_vec = matrix[word_map['love']]
    hate_vec = matrix[word_map['hate']]

    similarity = cossim(hate_vec, love_vec)
    print("cosine similarity('hate', 'love')", similarity)

    similarity = cossim(mean_vec1, love_vec)
    print("cosine similarity(text 1 mean, 'love')", similarity)
    similarity = cossim(mean_vec1, hate_vec)
    print("cosine similarity(text 1 mean, 'hate')", similarity)

    similarity = cossim(mean_vec2, love_vec)
    print("cosine similarity(text 2 mean, 'love')", similarity)
    similarity = cossim(mean_vec2, hate_vec)
    print("cosine similarity(text 2 mean, 'hate')", similarity)


main()
