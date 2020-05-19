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


def search_similar(word_map, matrix, query, threshold=0.95):
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
