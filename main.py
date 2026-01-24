import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import random
from treelib.tree import Tree

import utils
import trie_lib
import visualize
import gaddag_lib

DICE_CSV_PATH = './assets/dice.csv'
TRIE_PKL_PATH = './assets/trie.pkl'
GADDAG_PKL_PATH = './assets/gaddag.pkl'

def test_gaddag():
    words = utils.load_words()
    gaddag = gaddag_lib.get_gaddag(words)
    tree = Tree()
    tree.create_node(tag='Root', identifier='root')

    gaddag_lib.add_nodes(tree, 'root', gaddag)
    tree.show()


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

    for letter in starter:
        letters.remove(letter)
    rack = Counter(letters)
    if Path(GADDAG_PKL_PATH).exists():
        with open(GADDAG_PKL_PATH, 'rb') as f:
            gaddag = pickle.load(f)
    else:
        gaddag = gaddag_lib.get_gaddag(words)
        with open(TRIE_PKL_PATH, 'wb') as f:
            pickle.dump(gaddag, f)
            
    results = gaddag_lib.find_second_word(starter, rack, gaddag)
    if results:
        i, (j, word) = results[0], results[1][0]
    print(f"Second word: {word}")
    for idx, char in enumerate(word):
        board[11 - j + idx, 11 - (len(starter)//2) + i] = char.upper()
    
    for i, letter in enumerate(word):
        if i == j: continue
        letters.remove(letter)

    adjacents = utils.find_anchors(board)
    visualize.plot_board(board, letters, adjacents)

if __name__ == '__main__':
    starter_word()
