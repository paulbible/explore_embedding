"""
A program to explore the parts of speech for a given text.
"""
import nltk
from nltk.corpus import stopwords

# download some dictionaries / prediction models,
# Uncomment to download, the comment again.
#
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def main():
    filename = '../data/lincoln_2nd_SOTU.txt'

    # Stop words built in to nltk
    #
    # stops = stopwords.words('english')
    # print(stops)

    with open(filename) as f:
        for line in f:
            print('------  Next line / paragraph ------')
            line = line.strip()
            tokens = list(nltk.word_tokenize(line))
            print(tokens)

            ## Optional to filter stopwords
            # for word in tokens:
            #     if word in stops:
            #         tokens.remove(word)

            # tag each work with its part of speech
            tagged = nltk.pos_tag(tokens)
            print(tagged)
            keep_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
            # check tag code meanings here: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
            for (word, tag) in tagged:
                print('word:', word)
                print(' |-- tag:', tag)
                input('$>')

                ## Keep only 1 kind of tag / part of speech
                # if tag in keep_tags:
                #     print()
                #     print('word:', word)
                #     print(' |-- tag:', tag)
                #     input('$>')







main()
