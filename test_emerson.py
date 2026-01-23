import pandas as pd
import random
import nltk
from collections import Counter
import csv

def load_words():
    """
    Loads nltk words that are valid for Q-less
    """
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    from nltk.corpus import words
    return [w for w in words.words() if w.islower() and len(w) >= 3 and len(w) <= 12 and w.isalpha()]

def load_dice() -> list[list[str]]:
    """
    Loads the Q-less dice.

    Returns
    -------
    list of list of str
        The possible letters for each die.
    """
    with open('dice.csv', newline='') as f:
        return [list(map(str.lower, row)) for row in csv.reader(f, delimiter=',', quotechar='|')]

def roll(dice: list[list[str]]) -> list[str]:
    """
    Picks 12 random letters from the Q-less dice.

    Parameters
    ----------
    dice : list of list of str
        The dice to roll.

    Returns
    -------
    list of str
        The resulting roll.
    """
    return [random.choice(die) for die in dice]

def get_valid_words(roll: list[str], words: list[str]) -> list[str]:
    """
    Get all valid words for a roll.

    Parameters
    ----------
    roll : list of str
        The letters in the roll.
    words : list of str
        The pool of words to consider.

    Returns
    -------
    list of str
        All possible words in Q-less.
    """
    letters_count = Counter(roll)
    valid_words = []
    for word in words:
        word_count = Counter(word)
        if word_count <= letters_count:
            valid_words.append(word)
    return valid_words

if __name__ == '__main__':
    dice = load_dice()
    letters = roll(dice)
    words = load_words()
    print(f'Rolled letters: {letters}')
    print(f'Total words: {len(words)}')

    valid_words = get_valid_words(letters, words)

    print(f'Found {len(valid_words)} valid starting words')
    lengths = [len(w) for w in valid_words]
    length_counts = Counter(lengths)
    for length in sorted(length_counts):
        print(f'Length {length:<10}  Count {length_counts[length]:<10}')


    