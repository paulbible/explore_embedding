"""
This is a program to summarize clusters to report over represented words.
"""

import option_helpers as opth
from collections import defaultdict


def create_options():
    # Create some options for the tool.
    options = opth.default_option_map_input_output()
    # modify default descriptions for input and output
    options['input']['description'] = 'A comma separated file of sentences as rows with clusters labeled.'
    options['input']['input_name'] = '<cluster_file>'
    options['output']['description'] = 'An output file name for count matrix data (CSV).'
    # add custom options
    options['list'] = {
        'order': 3,
        'short': 'l',
        'long': 'list',
        'input_name': '<list>',
        'description': 'A list of names to order the output. These are a text file a file names.',
        'optional': True
    }
    return options


def read_names(filename):
    names = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            names.append(line)
    return names


def main():
    options = create_options()
    program_description = 'Process a table of sentences with cluster ids and speeches\n' \
                          '     into a count matrix for regression analysis.'
    print_usage_func = opth.print_usage_maker(program_description, options)
    parse_function = opth.parse_options_maker(options, print_usage_func)

    argument_map = parse_function()

    input_file = opth.validate_required('input', argument_map, print_usage_func)
    output_file = opth.validate_required('output', argument_map, print_usage_func)
    name_list = opth.with_default('list', argument_map, None)

    print(input_file, output_file)

    clusters = set()
    speeches = []
    speech_cluster_counts = {}

    with open(input_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'Sentence,cluster,file':
                # skip header
                continue
            # print(line)

            # split sentence record
            sentence, cluster, filename = line.split(',')

            # initialize cluster database
            if filename not in speech_cluster_counts:
                speech_cluster_counts[filename] = defaultdict(int)

            # add the cluster
            clusters.add(cluster)

            # add cluster count to speech
            speech_cluster_counts[filename][cluster] += 1

    clusters = list(clusters)
    clusters.remove('')
    cluster_ids = [int(cluster) for cluster in clusters]
    cluster_ids.sort()

    if name_list:
        filenames = read_names(name_list)
    else:
        filenames = list(speech_cluster_counts.keys())
        filenames.remove('')


    with open(output_file, 'w') as f:
        header = ['speech']
        for c in cluster_ids:
            header.append('c_' + str(c))

        # print(','.join(header))
        f.write(','.join(header) + '\n')

        for filename in filenames:
            # print(filename)
            cluster_counts = speech_cluster_counts[filename]
            outputs = [filename]
            for c in cluster_ids:
                if str(c) not in cluster_counts:
                    outputs.append('0')
                else:
                    outputs.append(str(cluster_counts[str(c)]))
            # print(','.join(outputs))
            f.write(','.join(outputs) + '\n')

main()
