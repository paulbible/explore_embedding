"""
    Simple example of clustering sentences
"""
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import nltk
from collections import defaultdict

import embedding_tools as et


def main():
    embedding_filename = '../data/glove.6B.50d.txt'
    word_map, matrix = et.load_embeddings(embedding_filename)
    print('embeddings loaded')

    # load text data from file.
    filename = '../data/lincoln_2nd_SOTU.txt'
    with open(filename) as f:
        data = f.read()

    # initialize sentence data
    sent_vectors = []

    # parse senteces from text
    sentences = nltk.sent_tokenize(data)
    # operate on each sentence one by one
    for sentence in sentences:
        # print(sentence)
        sent_vec = et.reduce_sum_word_list(sentence, word_map, matrix)
        # print(sent_vec)
        sent_vectors.append(sent_vec)
        # input()

    sent_mat = np.array(sent_vectors)
    nrows, ncols = sent_mat.shape
    print(sent_mat.shape)

    # decent tutorial
    # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    Y = pdist(sent_mat, 'cosine') # define distance between points (sentence vectors)
    Z = linkage(Y, 'ward') # define linkage, how to group points together

    # display the resulting clustering of al sentences
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
    plt.show()

    # cut the hierarchy at a level to make k clusters
    k = 10
    c_labels = fcluster(Z, k, criterion='maxclust')
    print(c_labels)

    # collect the sentence id for each cluster
    cluster_members = defaultdict(list)
    for i in range(nrows):
        # append sentece ID to the correct member list
        cluster_members[c_labels[i]].append(i)

    for i in range(1, k+1):
        print("***** cluster: %s, size: %s *****" % (i, len(cluster_members[i])) )
        for sent_id in cluster_members[i]:
            print("cluster %s sentence: " % i, sentences[sent_id])
        input()











main()
