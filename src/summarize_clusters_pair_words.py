"""
This is a program to summarize clusters to report over represented words.
"""

import option_helpers as opth
import nltk
import string
from nltk.corpus import stopwords
from collections import defaultdict


def create_options():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'A comma separated file of sentences as rows with clusters labeled.'
    options['input']['input_name'] = '<cluster_file>'
    options['output']['description'] = 'An output file name for the word distribution data (CSV).'
    # add custom options
    options['topN'] = {
        'order': 3,
        'short': 'n',
        'long': 'topN',
        'input_name': '<number>',
        'description': 'Return the N word pairs with the highest count for each cluster.',
        'optional': True
    }
    options['cutoff'] = {
        'order': 4,
        'short': 'c',
        'long': 'cutoff',
        'input_name': '<count_limit>',
        'description': 'Filter word pairs with less that <count_limit> occurrences. (default: 5)',
        'optional': True
    }
    options['percent'] = {
        'order': 5,
        'short': 'p',
        'long': 'percent',
        'input_name': '<percent_limit>',
        'description': 'Filter word pairs that account for less than <percent_limit> of cluster words (default: 0.0,all)',
        'optional': True
    }
    return options


def shared_top_10(cluster_word_database):
    top_10_set = set()

    for cluster in cluster_word_database:
        word_map = cluster_word_database[cluster]
        for k, v in sorted(word_map.items(), key=lambda item: item[1], reverse=True)[:min(10, len(word_map))]:
            top_10_set.add(k)

    return top_10_set


def main():
    options = create_options()
    program_description = 'Summarize the word distributions of sentence clusters\n' \
                          '    using frequent word pairs.'
    print_usage_func = opth.print_usage_maker(program_description, options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    argument_map = parse_function()

    input_file = opth.validate_required('input', argument_map, print_usage_func)
    output_file = opth.validate_required('output', argument_map, print_usage_func)
    count_limit = opth.with_default_int('cutoff', argument_map, 1)
    percent_limit = opth.with_default_float('percent', argument_map, 0.0)

    top_n = opth.with_default_int('topN', argument_map, None)

    # print(input_file, output_file, count_limit, percent_limit, top_n)

    stops = stopwords.words('english')
    # print(stops)
    # stops.extend(['america', 'americans'])
    # print(stops)

    cluster_word_database = {}
    cluster_word_total = defaultdict(int)

    all_data_word_counts = defaultdict(int)
    total_words = 0

    pair_word_noise = {"ca_n\'t", "wo_n't"}

    with open(input_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'Sentence,cluster,file':
                # skip header
                continue
            # print(line)

            # split sentence record
            sentence, cluster, filename = line.split(',')
            if cluster == '':
                continue

            # initialize cluster database
            if cluster not in cluster_word_database:
                cluster_word_database[cluster] = defaultdict(int)

            word_list = nltk.word_tokenize(sentence)
            word_count = len(word_list)
            # skip if there are not pairs in the sentence
            if word_count < 2:
                continue

            for i in range(word_count-2):
                word_1 = word_list[i].lower()
                word_2 = word_list[i+1].lower()
                word_1 = word_1.strip(string.punctuation)
                word_2 = word_2.strip(string.punctuation)
                if len(word_1) == 0 or word_1 in stops:
                    continue

                if len(word_2) == 0 or word_2 in stops:
                    continue

                pair_key = word_1 + '_' + word_2

                if pair_key in pair_word_noise:
                    continue

                cluster_word_database[cluster][pair_key] += 1
                cluster_word_total[cluster] += 1
                all_data_word_counts[pair_key] += 1
                total_words += 1

    ################
    # main filtering
    header = ['cluster', 'word', 'count', 'cluster_freq', 'total_freq']
    output_rows = []

    for cluster in cluster_word_total:
        total = float(cluster_word_total[cluster])
        # print(cluster, total)
        word_map = cluster_word_database[cluster]
        # print(word_map)

        # sort all cluster work pairs
        sorted_pairs = sorted(word_map.items(), key=lambda item: item[1], reverse=True)

        if top_n:
            limit = min(top_n, len(word_map))
        else:
            limit = min(10, len(word_map))

        count = 0
        for k, v in sorted_pairs:
            cluster_frequency = v/total
            dataset_frequency = all_data_word_counts[k]/total_words



            if v < count_limit:
                continue

            if cluster_frequency < percent_limit:
                continue

            outs = [cluster, k, str(v), str(round(cluster_frequency, 5)),
                    str(round(dataset_frequency, 5))]
            # print('\t'.join(outs))
            output_rows.append(outs)
            count += 1
            if count >= limit:
                break

        # input('press enter for next cluster')

    with open(output_file,'w') as f:
        f.write(','.join(header) + '\n')
        for row in output_rows:
            # print(','.join(row))
            f.write(','.join(row) + '\n')


main()
