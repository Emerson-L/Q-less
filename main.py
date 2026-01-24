import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import random

import utils
import trie_lib
import visualize

DICE_CSV_PATH = './assets/dice.csv'
TRIE_PKL_PATH = './assets/trie.pkl'

def test_trie():
    words = utils.load_words()
    if Path(TRIE_PKL_PATH).exists():
        with open(TRIE_PKL_PATH, 'rb') as f:
            trie = pickle.load(f)
    else:
        trie = trie_lib.generate_trie(words)
        with open(TRIE_PKL_PATH, 'wb') as f:
            pickle.dump(trie, f)


    words_test = ['aardvark', 'blarf', 'poop', 'fantastic']
    for word in words_test:
        print(f'{word} is {trie_lib.check_word(word, trie)}')


def starter_word():
    dice = utils.load_dice(DICE_CSV_PATH)
    letters = utils.roll(dice)
    words = utils.load_words()
    print(f'Rolled letters: {letters}')
    #print(f'Total words: {len(words)}')

    valid_words = utils.get_valid_words(letters, words)

    print(f'Found {len(valid_words)} valid starting words')

    # Print the counts of each lengths of word
    # lengths = [len(w) for w in valid_words]
    # length_counts = Counter(lengths)
    # for length in sorted(length_counts):
    #     print(f'Length {length:<5}  Count {length_counts[length]:<5}')

    starter = random.choice(valid_words)
    print(f'Starting word: {starter}')
    board = np.full((23, 23), '', dtype='<U1')
    
    for idx, char in enumerate(starter):
        board[11, 11 - (len(starter)//2) + idx] = char.upper()

    adjacents = utils.find_anchors(board)
    visualize.plot_board(board, letters, adjacents)

if __name__ == '__main__':
    #test_trie()
    starter_word()
