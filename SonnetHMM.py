##################################################
# Child class inheriting form HiddenMarkovModel  #
##################################################

import random
import numpy as np
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

    def train_model(sonnets, n_states, n_iters, n_iters_aux=0):
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
        for s in sonnets:
            X.append(le.transform([word for line in s for word in line]))

        X_lines = []
        for line in lines:
            X_lines.append(le.transform(line))


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

        # Train an auxillary model to learn best words for starting
        if n_iters_aux == 0:
            n_iters_aux = int(n_iters / 10)
        sm_aux = SonnetModel(A, O, A_start, le)
        sm_aux.unsupervised_learning(X_lines, n_iters)

        sm.aux_model = sm_aux

        sm.A = np.array(sm.A)
        sm.O = np.array(sm.O)

        return sm

    train_model = staticmethod(train_model)

    def increase_weight(self, wordlist, mult=10):
        '''
        Increase the probability of observing particular words
        '''
        wordlist = self.le.transform(wordlist)
        self.O = np.array(self.O)
        for word in wordlist:
            self.O[:, word] *= mult

        # normalize
        for j in range(self.L):
            self.O[j] /= np.sum(self.O[j])

    def generate_sonnet(self):
        '''
        Sonnet generator
        '''
        sonnet = []
        num_lines = 14
        rhymes = [None for _ in range(num_lines)]

        for i in range(num_lines):
            sonnet.append(list())

            # Determine if the word to rhyme with (if any)
            rhyme = None
            if i in [2, 3, 6, 7, 10, 11]:
                rhyme = rhymes[i - 2]
            elif i == 13:
                rhyme = rhymes[i - 1]

            # Choose initial state from auxillary model
            y_start = self.sample(self.aux_model.A_start)
            # Generate first word
            x_start = self.sample(self.aux_model.O[y_start])
            word = self.le.inverse_transform(x_start)

            # Add word
            sonnet[-1].append(word)
            j = self.get_syllable_count(word)

            # Transform to main model hidden states
            dist = [self.aux_model.O[k][x_start] for k in range(self.L)]
            norm = sum(dist)
            for k in range(len(dist)):
                dist[k] /= norm
            y = self.sample(dist)

            # Get next state
            y_next = self.sample(self.A[y])
            y = y_next

            num_sylls = 10
            while j < num_sylls:
                # -------- Get Word -------- #
                # Make observation
                x = self.sample(self.O[y])
                # Convert observation to word
                word = self.le.inverse_transform(x)

                # -------- Add Word -------- #
                # Check word validity
                allowed, word_syll, word_pr = self.check_word(word, j,
                                                              num_sylls, rhyme)
                if allowed:
                    sonnet[-1].append(word)
                    # Add to number of syllables
                    j += word_syll

                    # Make transition
                    y_next = self.sample(self.A[y])
                    y = y_next
                else:
                    continue
            rhymes[i] = word_pr

        return self.format_sonnet(sonnet)

    def format_sonnet(self, arr):
        '''
        Converts a sonnet from array format to a properly formatted one
        '''
        sonnet = ''
        num_lines = 14
        for i in range(num_lines):
            # Indent the couplet
            if i >= 12:
                sonnet += '  '

            for j in range(len(arr[i])):
                word = arr[i][j]
                if (j == 0 or word == 'i' or word == "i'll" or word == "i'm" or
                        word == 'o'):
                    sonnet += word[0].upper() + word[1:]
                else:
                    sonnet += word

                if j < len(arr[i]) - 1:
                    sonnet += ' '

            sonnet += '\n'
        return sonnet

    def check_word(self, word, i, num_sylls, rhyme_pr=None):
        '''
        Checks if a word is allowed to be placed at syllable i
        '''
        pronunciations = self.get_syllable_stress(word)

        for pronunciation in pronunciations:
            word_syll = pronunciation[0]
            word_stress = pronunciation[1]
            word_pr = pronunciation[2]
            # If we exceed syllable count, resample
            if i + word_syll > num_sylls:
                continue

            # Check rhyme if necessary
            if rhyme_pr is not None and i + word_syll == num_sylls:
                # Only allow words with known rhymes
                if word_pr is None:
                    continue

                # Permissive rhyming scheme: either last vowel or last
                # consonant sound must be the same

                word_last_vowel = None
                rhyme_last_vowel = None
                for pr in word_pr:
                    if pr[-1].isdigit():
                        word_last_vowel = pr
                for pr in rhyme_pr:
                    if rhyme_pr[-1].isdigit():
                        rhyme_last_vowel = pr

                rhyme = False
                if word_last_vowel == rhyme_last_vowel or \
                        word_pr[-1] == rhyme_pr[-1]:
                    rhyme = True

                if not rhyme:
                    continue

            # Check stress patterns for words  > 1 syllable
            if len(word_stress) > 1:
                for j in range(len(word_stress)):
                    # Don't allow stressed syllables on odd syllables
                    if (i + j + 1) % 2 == 1 and word_stress[j] == 1:
                        continue
                    # Avoid unstressed syllables on odd syllables
                    if (i + j + 1) % 2 == 0 and word_stress[j] == 0:
                        P_resample = 0.5
                        rand = random.uniform(0., 1.)
                        if rand < P_resample:
                            continue

            return True, word_syll, word_pr

        return False, 0, None

    def get_syllable_count(self, word):
        '''
        Use the CMU pronunciation dictionary to get a syllable count.

        Returns:
            num_sylls (int): Number of syllables
        '''
        num_sylls = 0

        if word in self.prdict:
            pronunciation = self.prdict[word][0]
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

    def get_syllable_stress(self, word):
        '''
        Use the CMU pronunciation dictionary to get a syllable count and stress
        patterns

        Returns:
            list of tuples for all pronunciations
                num_sylls (int): Number of syllables
                stress (list): List of length num_sylls, showing stress
                               patterns
        '''
        pr_list = []

        if word in self.prdict:
            for pronunciation in self.prdict[word]:
                num_sylls = 0
                stress = []
                for phoneme in pronunciation:
                    if phoneme[-1].isdigit():
                        num_sylls += 1
                        if int(phoneme[-1]) == 1:
                            stress.append(1)
                        else:
                            stress.append(0)

                pr_list.append((num_sylls, stress, pronunciation))
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
            stress = [0 for _ in range(num_sylls)]

            pr_list.append((num_sylls, stress, None))

        return pr_list

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

        assert np.isclose(sum(dist), 1), "Probability does not sum to one"

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
        line = line.lstrip(' ').rstrip('\n').rstrip(' ')

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
                # print(words)
                for word in words:
                    # Check if word begins with a paranthesis
                    if word[0] == '(' or word[0] == "'":
                        sonnets[-1][-1].append(word[1:].lower())
                    # Check if word ends in a punctuation
                    elif word[-1].isalnum():
                        sonnets[-1][-1].append(word.lower())
                    else:
                        sonnets[-1][-1].append(word[:-1].lower())
                        # For now, ignore punctuation
    #                     sonnets[-1][-1].append(word[-1])
    return sonnets
