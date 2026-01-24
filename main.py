import pickle
from pathlib import Path
from collections import Counter

import utils
import trie_lib


def test_trie():
    words = utils.load_words()
    if Path('trie.pkl').exists():
        with open('trie.pkl', 'rb') as f:
            trie = pickle.load(f)
    else:
        trie = trie_lib.generate_trie(words)
        with open('trie.pkl', 'wb') as f:
            pickle.dump(trie, f)


    words_test = ['aardvark', 'blarf', 'poop', 'fantastic']
    for word in words_test:
        print(f'{word} is {trie_lib.check_word(word, trie)}')


def starter_word():
    dice = utils.load_dice()
    letters = utils.roll(dice)
    words = utils.load_words()
    print(f'Rolled letters: {letters}')
    print(f'Total words: {len(words)}')

    valid_words = utils.get_valid_words(letters, words)

    print(f'Found {len(valid_words)} valid starting words')
    lengths = [len(w) for w in valid_words]
    length_counts = Counter(lengths)
    for length in sorted(length_counts):
        print(f'Length {length:<5}  Count {length_counts[length]:<5}')

if __name__ == '__main__':
    #test_trie()
    starter_word()
