import math
import argparse
import codecs
from collections import defaultdict
import random
from numpy import random

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

        # The uniform distribution in the corpus
        self.index_list = []
        self.unigram_distribution_list = []


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # YOUR CODE HERE

                for index in range(self.unique_words):
                    i, word, number = f.readline().strip().split(' ')
                    i = int(i)
                    number = int(number)
                    self.word[i] = word
                    self.index[word] = i
                    self.unigram_count[word] = number
                    self.index_list.append(i)
                    self.unigram_distribution_list.append(number/self.total_words)
                
                row = f.readline().strip().split(' ')
                while row != ["-1"]:                   
                    i1 = self.word[int(row[0])]
                    i2 = self.word[int(row[1])]
                    prob = math.exp(float(row[2]))
                    self.bigram_prob[i1][i2] = prob
                    row = f.readline().strip().split(' ')
                     
                # END OF MY CODE

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """ 
        # YOUR CODE HERE

        sentence = ""
        current_word = w
        sentence = sentence + " " + current_word
        possible_words = []
        probabilities = []

        
        for step in range(n):
            #print(f'step: {step} current word: {current_word} index: {self.index[current_word]} sentence: {sentence}')

            if current_word in self.bigram_prob:
                bigrams = self.bigram_prob[current_word]
                for option in bigrams:
                    possible_words.append(option)
                    probabilities.append(bigrams[option])
                random_word_array = random.choice(possible_words, 1, probabilities)
                random_word = random_word_array.item()                
                possible_words = []
                probabilities = []
                            
            else:
                #print ("Dags for a random word")
                random_word_index_array = random.choice(self.index_list, 1, self.unigram_distribution_list)  # choose any word according to unigram distribution
                random_word_index = random_word_index_array.item()
                random_word = self.word[random_word_index]
                
            current_word = random_word
            sentence = sentence + " " + current_word
            
        print(sentence)
        
def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
