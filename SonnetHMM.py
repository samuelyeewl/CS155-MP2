##################################################
# Child class inheriting form HiddenMarkovModel  #
##################################################

import random
from HMM import HiddenMarkovModel
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import cmudict

class SonnetModel(HiddenMarkovModel):
    '''
    Sonnet Class to generate a sonnet using HMM
    '''
    ### Common Dictionary
    prdict = cmudict.dict()

    def __init__(self, A, O, A_start, le):
        '''
        Initialize SonnetModel class

        Args:
            A: transition matrix
            O: observation matrix
            le: LabelEncoder
        '''
        super().__init__(A, O)
        self.le = le
        self.A_start = A_start

    def train_model(sonnets, n_states, n_iters):
        '''
        Factory method to train a SonnetModel
        '''
        # Flatten data
        lines = [line for s in sonnets for line in s]
        words = [word for line in lines for word in line]

        # Encode data into X matrix
        le = LabelEncoder()
        le.fit(words)
        X = []
        for line in lines:
            X.append(le.transform(line))

        # Train unsupervised model
        # Make a set of observations.
        observations = set()
        for x in X:
            observations |= set(x)

        # Compute L and D.
        L = n_states
        D = len(observations)

        # Randomly initialize and normalize matrices A and O.
        A = [[random.random() for i in range(L)] for j in range(L)]

        for i in range(len(A)):
            norm = sum(A[i])
            for j in range(len(A[i])):
                A[i][j] /= norm

        A_start = [random.random() for i in range(L)]
        norm = sum(A_start)
        for i in range(len(A_start)):
            A_start[i] /= norm

        O = [[random.random() for i in range(D)] for j in range(L)]

        for i in range(len(O)):
            norm = sum(O[i])
            for j in range(len(O[i])):
                O[i][j] /= norm

        sm = SonnetModel(A, O, A_start, le)
        sm.unsupervised_learning(X, n_iters)

        return sm

    train_model = staticmethod(train_model)

    def generate_sonnet(self):
        '''
        Sonnet generator
        '''
        sonnet = ''
        num_lines = 14
        for i in range(num_lines):
            # Indent the couplet
            if i >= 12:
                sonnet += '  '

            # Choose initial state
            y = self.sample(self.A_start)

            num_sylls = 10
            i = 0
            while i < num_sylls:
                print(i)
                # -------- Get Word -------- #
                # Make observation
                x = self.sample(self.O[y])
                # Convert observation to word
                word = self.le.inverse_transform(x)

                # -------- Check Word ------ #
                word_syll = self.get_syllable_count(word)
                # If we exceed syllable count, resample
                if i + word_syll > num_sylls:
                    print(i, word_syll)
                    continue

                # -------- Add Word -------- #
                # Capitalize first word
                if i == 0 or word == 'i':
                    word = word[0].upper() + word[1:]
                sonnet += word
                # Add to number of syllables
                i += word_syll
                # Add space
                if i < num_sylls:
                    sonnet += ' '

                # Make transition
                y_next = self.sample(self.A[y])
                y = y_next

            sonnet += '\n'

        return sonnet

    def get_syllable_count(self, word):
        '''
        Use the CMU pronunciation dictionary to get a syllable count
        '''
        if word in self.prdict:
            pronunciation = self.prdict[word][0]
            num_sylls = 0
            for phoneme in pronunciation:
                if phoneme[-1].isdigit():
                    num_sylls += 1
        else:
            # If the word is not in the dictionary, use a heuristic
            # of counting groups of consecutive vowels
            num_sylls = 0
            consec_vowel = False
            for i in range(len(word)):
                # Check for vowel
                if len(word[i].lstrip('aeiouy')) == 0:
                    if not consec_vowel:
                        num_sylls += 1
                        consec_vowel = True
                else:
                    consec_vowel = False

        return num_sylls

    def sample(self, dist):
        '''
        Sample from a distribution

        Args:
            dist (np.ndarray): Probability distribution
        Returns:
            index chosen
        '''
        # Generate a random number
        num = random.uniform(0., 1.)

        for i in range(len(dist)):
            num -= dist[i]
            if num < 0:
                return i


def read_data(filename):
    '''
    Function to read in poems from a text file.
    Returns a 3D array organized by sonnet, line, word
    '''
    with open(filename) as f:
        data = f.readlines()

    in_sonnet = False
    sonnets = []
    for line in data:
        # Strip whitespace from ends
        line = line.lstrip(' ').rstrip('\n')

        # Check if start of sonnet
        if line.isnumeric():
            in_sonnet = True
            sonnets.append(list())
        elif in_sonnet:
            # Check if end of sonnet
            if len(line) == 0:
                in_sonnet = False
            # Otherwise add line to sonnet
            else:
                sonnets[-1].append(list())
                # Split on whitespace
                words = line.split(' ')

                for word in words:
                    # Check if word begins with a paranthesis
                    if word[0] == '(':
                        sonnets[-1][-1].append(word[1:].lower())
                    # Check if word ends in a punctuation
                    elif word[-1].isalnum():
                        sonnets[-1][-1].append(word.lower())
                    else:
                        sonnets[-1][-1].append(word[:-1].lower())
                        # For now, ignore punctuation
    #                     sonnets[-1][-1].append(word[-1])
    return sonnets
