########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
#from Utility import Utility
import preprocess

def unsupervised_learning(sequence, n_states, n_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #genres, genre_map = Utility.load_ron_hidden()
    #genres = preprocess.preprocess_word_to_num('shakespeare.txt')
    # print genres
    # Train the HMM.
    HMM = unsupervised_HMM(sequence, n_states, n_iters)

    # Print the transition matrix.
    '''
    print("Transition Matrix:")
    print('#' * 70)
    for i in range(len(HMM.A)):
        print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    print('')
    print('')

    # Print the observation matrix. 
    print("Observation Matrix:  ")
    print('#' * 70)
    for i in range(len(HMM.O)):
        print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    print('')
    print('')
    '''
    return HMM.A, HMM.O


if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code Unsupervise Learning"))
    print('#' * 70)
    print('')
    print('')

    unsupervised_learning(10, 2)
