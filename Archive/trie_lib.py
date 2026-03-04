import utils
import json
from pathlib import Path
import pickle

import config


def generate_trie(words:list, visualize:bool=False) -> dict:
    """
    Indy write this
    """
    trie_root = {}
    for i, word in enumerate(words):
        trie = trie_root
        for j, letter in enumerate(word):
            if letter not in trie:
                trie[letter] = {}
            trie = trie[letter]
            if j + 1 == len(word):
                trie['*'] = True

        if visualize:
            if i == 5: 
                with open('./assets/trie.json', 'w') as f:
                    f.write(json.dumps(trie_root, indent=4))
                break

    return trie_root

def check_word(word:str, trie:dict) -> bool:
    """
    Indy write this
    """
    for letter in word:
        if letter in trie:
            trie = trie[letter]
        else:
            return False
    return '*' in trie


def test_trie():
    words = utils.load_words()
    if Path(config.TRIE_PKL_PATH).exists():
        with open(config.TRIE_PKL_PATH, 'rb') as f:
            trie = pickle.load(f)
    else:
        trie = generate_trie(words)
        with open(config.TRIE_PKL_PATH, 'wb') as f:
            pickle.dump(trie, f)


    words_test = ['aardvark', 'blarf', 'poop', 'fantastic']
    for word in words_test:
        print(f'{word} is {check_word(word, trie)}')
