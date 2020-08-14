"""
    This program provides options to cluster the sentences of a raw text corpus.
"""
import option_helpers as opth
import string
import os
import embedding_tools as et
import nltk
import math
from nltk.corpus import stopwords
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict

keep_tags = {'NN', 'NNS', 'NNP', 'NNPS',
             'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'JJ', 'JJR', 'JJS',
             'RB', 'RBR', 'RBS'}


def create_options():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'An input folder with each document represented as a single text file.'
    options['input']['input_name'] = '<input_folder>'
    options['output']['description'] = 'An output file name for the clustered sentence table.'
    # add custom options
    options['embedding'] = {
        'order': 3,
        'short': 'e',
        'long': 'embedding',
        'input_name': '<embedding_file>',
        'description': 'The file containing the vector space embedded vocabulary.',
        'optional': False
    }
    options['num_clusters'] = {
        'order': 4,
        'short': 'k',
        'long': 'num_clusters',
        'input_name': '<num_clusters>',
        'description': 'The number of sentence clusters to partition the sentences into.',
        'optional': False
    }
    options['stopfilter'] = {
        'order': 5,
        'short': 's',
        'long': 'stopfilter',
        'input_name': None,
        'description': 'Filter out stopwords (a, an ,the, etc.) before clustering',
        'optional': True
    }
    options['pos'] = {
        'order': 6,
        'short': 'p',
        'long': 'pos',
        'input_name': None,
        'description': 'Keep only words that are a major part of speech (Nouns, Verbs, Adjectives, and Adverbs).',
        'optional': True
    }
    options['tfidf'] = {
        'order': 6,
        'short': 't',
        'long': 'tfidf',
        'input_name': None,
        'description': 'Apply an inverse document frequency weighting. log(#sentences/#setnences with word)',
        'optional': True
    }
    return options


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

def calculate_sentence_vectors_tfidf(input_folder, word_map, matrix, do_stop_filtering=False, do_pos_filtering=False):

    sentence_db_temp = []
    file_labels_temp = []

    idf_counts = defaultdict(int)
    # Document frequency loop
    files_to_process = os.listdir(input_folder)
    for filename in files_to_process:
        full_filename = os.path.join(input_folder, filename)
        with open(full_filename, encoding='utf-8', errors='ignore') as f:
            data = f.read()
            # get all sentences form the file
            current_sentences = nltk.sent_tokenize(data)

            for sentence in current_sentences:
                sentence_db_temp.append(sentence)
                file_labels_temp.append(filename)

                # remove newlines
                sentence = sentence.replace('\n', ' ')
                # remove trailing punctuation.
                sentence.strip(string.punctuation)

                word_list = nltk.word_tokenize(sentence)
                for word in set(word_list):
                    # add 1 for each sentence where the sentence is found.
                    idf_counts[word.lower()] += 1

    idf_weights = {}
    doc_count = len(sentence_db_temp)
    for word in idf_counts:
        # create a weight for very word that reflects how many sentences it appears in
        # Words appearing in nearly every sentence will be near 0. Words that are uncommon get higher weighting.
        idf_weights[word] = math.log(doc_count/idf_counts[word])

    # print(len(sentence_db_temp))
    # print('we', idf_counts['we'], idf_weights['we'])
    # print('i', idf_counts['i'], idf_weights['i'])
    # print('a', idf_counts['a'], idf_weights['a'])
    # print('an', idf_counts['an'], idf_weights['an'])
    # print('the', idf_counts['the'], idf_weights['the'])
    # print('america', idf_counts['america'], idf_weights['america'])
    # print('american', idf_counts['american'], idf_weights['american'])
    # print('people', idf_counts['people'], idf_weights['people'])
    # print('health', idf_counts['health'], idf_weights['health'])
    # print('war', idf_counts['war'], idf_weights['war'])

    # second loop
    sent_vectors_database = []
    sentences_database = []
    # these file names will help identify sentences
    file_name_labels = []

    # for each sentence
    for i in range(doc_count):
        sentence = sentence_db_temp[i]
        sentence_label = file_labels_temp[i]

        sentence = sentence.replace('\n', ' ')
        # remove trailing punctuation.
        sentence.strip(string.punctuation)

        word_list = nltk.word_tokenize(sentence)
        # conditionally apply word filtering for sentences
        if do_stop_filtering:
            word_list = filter_stopwords_list_to_list(word_list)

        if do_pos_filtering:
            word_list = filter_pos_list_to_list(word_list, keep_tags)

        # discard any empty sentences (skip them)
        if len(word_list) <= 0:
            continue

        # caculate word vector
        sent_vec = et.reduce_sum_word_list_weighted(word_list, word_map, matrix, idf_weights)
        # add the transformed data to the data storage variables
        sentences_database.append(sentence)
        sent_vectors_database.append(sent_vec)
        # add the filename to a list too
        file_name_labels.append(sentence_label)

    return sent_vectors_database, sentences_database, file_name_labels


def calculate_sentence_vectors(input_folder, word_map, matrix, do_stop_filtering=False, do_pos_filtering=False):
    sent_vectors_database = []
    sentences_database = []
    # these file names will help identify sentences
    file_name_labels = []

    files_to_process = os.listdir(input_folder)
    for filename in files_to_process:
        # get full path name and process
        full_filename = os.path.join(input_folder, filename)
        with open(full_filename, encoding='utf-8', errors='ignore') as f:
            data = f.read()
            # get all sentences form the file
            current_sentences = nltk.sent_tokenize(data)

            for sentence in current_sentences:
                # remove newlines
                sentence = sentence.replace('\n', ' ')
                # remove trailing punctuation.
                sentence.strip(string.punctuation)

                word_list = nltk.word_tokenize(sentence)
                # conditionally apply word filtering for sentences
                if do_stop_filtering:
                    word_list = filter_stopwords_list_to_list(word_list)

                if do_pos_filtering:
                    word_list = filter_pos_list_to_list(word_list, keep_tags)

                # discard any empty sentences (skip them)
                if len(word_list) <= 0:
                    continue

                # caculate word vector
                sent_vec = et.reduce_sum_word_list(word_list, word_map, matrix)
                # add the transformed data to the data storage variables
                sentences_database.append(sentence)
                sent_vectors_database.append(sent_vec)
                # add the filename to a list too
                file_name_labels.append(filename)

    return sent_vectors_database, sentences_database, file_name_labels


def main():
    options = create_options()
    print_usage_func = opth.print_usage_maker('This is a tool for clustering text in a corpus.', options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    argument_map = parse_function()

    input_folder = opth.validate_required('input', argument_map, print_usage_func)
    output_filename = opth.validate_required('output', argument_map, print_usage_func)
    embedding_filename = opth.validate_required('embedding', argument_map, print_usage_func)
    num_clusters = opth.validate_required_int('num_clusters', argument_map, print_usage_func)
    do_stop_filtering = opth.has_option('stopfilter', argument_map)
    do_pos_filtering = opth.has_option('pos', argument_map)
    do_tfidf_weighting = opth.has_option('tfidf', argument_map)

    # print(input_folder)
    # print(output_filename)
    # print(embedding_filename)
    # print(num_clusters)
    # print('do stop', do_stop_filtering)
    # print('do pos', do_pos_filtering)

    print('Loading Embedding Vectors ...')
    word_map, matrix = et.load_embeddings(embedding_filename)
    print('Done')

    print('Processing input files and making sentence database ...')
    if do_stop_filtering:
        print('with stop word filtering ...')

    if do_pos_filtering:
        print('with part-of-speech filtering')

    if do_tfidf_weighting:
        print('with inverse document (sentence) frequency weighting')
        # Call the function to calculate the word vectors
        sent_vectors_database, sentences_database, file_name_labels = calculate_sentence_vectors_tfidf(
            input_folder, word_map, matrix, do_stop_filtering, do_pos_filtering)
    else:
        # Call the function to calculate the word vectors
        sent_vectors_database, sentences_database, file_name_labels = calculate_sentence_vectors(
            input_folder, word_map, matrix, do_stop_filtering, do_pos_filtering)
    print('Done')

    print('Caclulating clusters  with k = %s ...' % str(num_clusters))
    sent_mat = np.array(sent_vectors_database)
    print("Vector Map Size:", sent_mat.shape)
    nrows, ncols = sent_mat.shape

    y = pdist(sent_mat, 'cosine')  # define distance between points (sentence vectors)
    z = linkage(y, 'ward')  # define linkage, how to group points together

    c_labels = fcluster(z, num_clusters, criterion='maxclust')
    print('Done')

    print('Writing clustered sentences to output table ...')
    # open the output file
    # one of the sentences has a unicode that causes problems. make the output utf-8 too.
    fout = open(output_filename, 'w', encoding='utf-8')
    header = ['Sentence', 'cluster', 'file']
    fout.write(','.join(header) + '\n')

    # collect the sentence id for each cluster, not needed right now.
    # cluster_members = defaultdict(list)
    for i in range(nrows):
        # sentences have commas, so take them out.
        sentence_string = sentences_database[i].replace(',', ' ')
        outputs = [sentence_string, str(c_labels[i]), file_name_labels[i]]
        # use 'join' to create the row string
        fout.write(','.join(outputs) + '\n')
    fout.close()
    print('Done')


main()
