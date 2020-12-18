"""
    A top level script to run all the anlaysis
"""
import subprocess
import os


def main():
    # Variables for first pass clustering
    n_clusters = 22
    output_dir = '../data/'
    sotu_speech_folder = '../data/speeches'
    speech_filename_list = '../data/names.txt'
    embedding_file = os.path.join(output_dir, 'glove.6B.50d.txt')

    R_command = 'C:\\Program Files\\R\\R-4.0.2\\bin\\x64\\Rscript.exe'

    # generated files
    sentence_cluster_out_file = os.path.join(output_dir, 'test_sent_clusters.csv')
    regression_matrix_file = os.path.join(output_dir,'regression_count_matrix.csv')

    # Run initial with clustering with cluster_text_tool.py
    args = ['python', 'cluster_text_tool.py','-i', sotu_speech_folder, '-o', sentence_cluster_out_file,
            '-e', embedding_file, '-k', str(n_clusters), '-s', '-t']
    print('## running: ', ' '.join(args))
    subprocess.run(args)

    # Create a summary of the clusters using the summarize_clusters_pair_words.py
    # '-i ../data/test_sent_table.csv -o ../data/pair_word_dist_test.csv'
    args = ['python', 'summarize_clusters_pair_words.py','-i', sentence_cluster_out_file,
            '-o', os.path.join(output_dir, 'pair_words_cluster_summary.csv')]
    subprocess.run(args)

    # Create a speech cluster count matrix
    # '-i ../data/test_sent_clusters.csv -o ../data/test_count_matrix.csv -l ../data/names.txt'
    args = ['python', 'prepare_regression_count_matrix.py',
            '-i', sentence_cluster_out_file, '-o', regression_matrix_file,
            '-l', speech_filename_list]
    subprocess.run(args)

    # Run the R resgression
    args = [R_command, 'sotu_regression.R']
    subprocess.run(args)


main()
