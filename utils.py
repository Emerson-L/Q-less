import numpy as np
import random
import nltk
from collections import Counter
import csv

def is_qless_word(word: str) -> bool:
    """
    Return if a word is a valid Q-less word or not.

    Parameters
    ----------
    word : str
        The word to check.

    Returns
    -------
    bool
        True if the word is valid and False otherwise.

    Notes
    -----
    A word is valid in Q-less if it is at least 3 letters. However, given
    that there are only 12 dice and the prior constraint, 11 and 13+ letter
    words aren't possible.
    """
    if word.islower() and word.isalpha() and len(word) <=12 and len(word) not in [1,2,11]:
        return True
    return False

def load_words():
    """
    Loads nltk words that are valid for Q-less

    Returns
    -------
    list of str
        All words in the nltk dataset
    """
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    from nltk.corpus import words
    return [w for w in words.words() if is_qless_word(w)]

def load_dice(dice_csv_path:str) -> list[list[str]]:
    """
    Loads the Q-less dice.

    Returns
    -------
    list of list of str
        The possible letters for each die.
    """
    with open(dice_csv_path, newline='') as f:
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


def find_anchors(board:np.ndarray) -> np.ndarray:
    """
    Find the anchors in a board and return an array of the coordinates of sqaures adjacent to those that are filled
    """
    filled = np.argwhere(board != '')
    adjacent = np.full((board.shape[0], board.shape[1]), False)
    for square in filled:
        right = (square[0], square[1] + 1)
        left = (square[0], square[1] - 1)
        top = (square[0] - 1, square[1])
        bottom = (square[0] + 1, square[1])
        for possible_adjacent in right, left, bottom, top:
            if not any((filled == possible_adjacent).all(axis=1)):
                adjacent[possible_adjacent] = True

    return np.argwhere(adjacent == True)

    