"""
    A program to load vectors from the GloVe work embeddings
"""
import numpy as np


def main():
    filename = '../data/glove.6B.50d.txt'
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
            # print("rest:", rest)
            # print("vector length:", len(rest))


    # get the number of rows and columns
    rows = len(matrix)
    cols = len(matrix[0])

    # create a numpy array (matrix)
    # rows = words
    # columns = encoding values, number of columns = number of dimensions
    matrix = np.array(matrix)

    # explore the word embeddings
    while True:
        test_word = input('>')
        if test_word in word_map:
            index = word_map[test_word]
            print(matrix[index])
        else:
            print("no word '%s' in the database" % test_word)


main()
