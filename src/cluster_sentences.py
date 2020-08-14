"""
    Simple example of clustering sentences
"""
import io
from collections import defaultdict
import string
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
# from matplotlib import pyplot as plt
# from pca.examples import results
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

import embedding_tools as et

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


def filter_pos_list_to_list(words, keep_tags):
    out_words = []
    tagged = nltk.pos_tag(words)
    for (word, tag) in tagged:
        if tag in keep_tags:
            out_words.append(word)
    return out_words


def filter_stopwords_list_to_list(words):
    out_words = []
    stops = stopwords.words('english')
    for word in words:
        if word not in stops:
            out_words.append(word)
    return out_words


def main():
    embedding_filename = '../data/glove.6B.50d.txt'
    word_map, matrix = et.load_embeddings(embedding_filename)
    # print('embeddings loaded')

    # data locations
    in_data_dir = '../data/speeches'
    out_data_dir = '../data/results'
    # summary_table_filename = 'sentences_by_cluster_all_nofilter.csv'
    summary_table_filename = 'sentences_by_cluster_NVJR_all.csv'

    # tags to keep for pos
    # starts N, noun
    # starts V, verb
    # starts J, adjective
    # starts R, adverb
    keep_tags = {'NN', 'NNS', 'NNP', 'NNPS',
                 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                 'JJ', 'JJR', 'JJS',
                 'RB', 'RBR', 'RBS'}

    # data storage variables
    sent_vectors_database = []
    sentences_database = []
    # these file names will help identify sentences
    file_name_labels = []

    '''
        ***** main sentence / data processing loop ***** 
    '''
    file_to_process = os.listdir(in_data_dir)
    for filename in file_to_process:
        # TESTING, if the file is not a Bush file, skip it.
        # if not filename.startswith('Bush'):
        #     continue

        # get the data file's path (so python can find it)
        full_filename = in_data_dir + os.path.sep + filename
        # open the file
        with open(full_filename, encoding='utf-8', errors='ignore') as f:

            data = f.read()
            # get all sentences form the file
            current_sentences = nltk.sent_tokenize(data)

            # this was your problem. you need to be careful about indentation.
            for sentence in current_sentences:
                # remove newlines
                sentence = sentence.replace('\n', ' ')
                # remove trailing punctuation.
                sentence.strip(string.punctuation)

                '''
                    ***** sentence filtering before vector conversion *****
                '''
                # add the sentence to the database
                word_list = nltk.word_tokenize(sentence)
                #word_list = filter_stopwords_list_to_list(word_list)
                #word_list = filter_pos_list_to_list(word_list, keep_tags)

                if len(word_list) <= 0:
                    continue


                # print(sentence)
                # print(word_list)
                # input()

                sent_vec = et.reduce_sum_word_list(word_list, word_map, matrix)
    
                # add the transformed data to the data storage variables
                sentences_database.append(sentence)
                sent_vectors_database.append(sent_vec)

                # add the filename to a list too
                file_name_labels.append(filename)

    # Summary
    print('sentences', len(sentences_database))
    print('file name labels', len(file_name_labels))
    print('sentence vectors', len(sent_vectors_database))

    # input()

    '''
        ***** Clustering ***** 
    '''
    # convert the table to a matrix
    sent_mat = np.array(sent_vectors_database)
    print("Vector Map Size:", sent_mat.shape)
    nrows, ncols = sent_mat.shape
    # decent tutorial
    # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial
    y = pdist(sent_mat, 'cosine')  # define distance between points (sentence vectors)
    z = linkage(y, 'ward')  # define linkage, how to group points together

    # display the resulting clustering of al sentences
    # dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
    # plt.show()
    #
    # cut the hierarchy at a level to make k clusters
    k = 12
    c_labels = fcluster(z, k, criterion='maxclust')
    print(c_labels)

    '''
        ***** output processing ***** 
    '''
    # open the output file
    summary_table_filename_path = (out_data_dir + os.path.sep + summary_table_filename)
    # one of the sentences has a unicode that causes problems. make the output utf-8 too.
    fout = open(summary_table_filename_path, 'w', encoding='utf-8')

    header = ['Sentence', 'cluster', 'file']
    fout.write(','.join(header) + '\n')

    # collect the sentence id for each cluster, not needed right now.
    # cluster_members = defaultdict(list)
    for i in range(nrows):
        # get the cluster and add it to the cluster map.
        sentence_i_cluster = c_labels[i]
        # cluster_members[sentence_i_cluster].append(i)

        # How I usually do output, create an output list of strings.
        # sentences have commas, so take them out.
        sentence_string = sentences_database[i].replace(',', ' ')
        outputs = [sentence_string, str(c_labels[i]), file_name_labels[i]]
        # use 'join' to create the row string
        fout.write( ','.join(outputs) + '\n')

    fout.close()
    print('done')


main()
