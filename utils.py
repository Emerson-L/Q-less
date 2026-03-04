import numpy as np
import random
from collections import Counter
import csv
from assets import twl

import config
import visualize

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
    words aren't possible. Also, no dice have `q` so words containing it are
    invalid.
    """
    conditions = [
        word.islower(),
        word.isalpha(),
        len(word) <= 12,
        len(word) not in [1, 2, 11],
        'q' not in word
    ]
    return all(conditions)

def load_words() -> list[str]:
    """
    Loads nltk words that are valid for Q-less

    Returns
    -------
    list of str
        All words in the nltk dataset
    """
    return [w for w in twl.iterator() if is_qless_word(w)]

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

def letters_to_numbers(letters:list[str]|str) -> list[int]:
    """
    Converts a list of characters or string of characters into their index in the alphabet.
    A/a = 0, B/b = 1, etc

    Parameters
    ----------
    letters : list of str or str
        List of single-character strings or single string of characters to convert.

    Returns
    -------
    nums : list[int]
        A list of integers corresponding to the given letters
    """
    return [ord(character.lower()) - 97 for character in letters]

def numbers_to_letters(nums: list[int]) -> list[str]:
    """
    Converts a list of 0-based indices into uppercase characters.
    0 = A, 1 = B, etc

        Parameters
    ----------
    nums : list of int
        List of numbers to convert

    Returns
    -------
    letters : list of str
        A list of characters corresponding to the given numbers 
    """
    return [chr(num + 97).upper() for num in nums]

def chars74k_sample_nums_to_characters(nums: list[int]) -> list[str]:
    """
    Converts sample numbers from the Chars74k dataset into their respective characters

    Parameters
    ----------
    nums: list of int
        sample numbers to convert
    
    Returns
    -------
    list of str
        characters that correspond with the sample numbers
    """
    result = []
    for n in nums:
        if 1 <= n <= 10:
            result.append(chr(n - 1 + ord('0')))
        elif 11 <= n <= 36:
            result.append(chr(n - 11 + ord('A')))
        elif 37 <= n <= 62:
            result.append(chr(n - 37 + ord('a')))
        else:
            raise ValueError('Number out of range')
    return result

def is_possible_roll(roll: list[str], dice: list[list[str]]) -> bool:
    """
    Find whether a given roll is possible from a set of dice using maximum bipartite matching.

    Parameters
    ----------
    roll : list of str
        The letters resulting from the potential roll.
    dice : list of list of str
        The sides of each die to compare against.

    Returns
    -------
    bool
        True if the roll is possible and false otherwise.
    """
    if len(roll) != len(dice):
        print("Roll length and dice length must match.")
        return False
    dice_sides = list(map(set, dice))
    G = [[False for _ in range(len(dice))] for _ in range(len(dice))]
    for i, letter in enumerate(roll):
        for j, sides in enumerate(dice_sides):
            if letter in sides:
                G[i][j] = True
    matches = [-1] * len(dice)
    for u in range(len(roll)):
        seen = [False] * len(dice)
        if not bipartite_match(G, u, seen, matches):
            return False
    #visualize.visualize_bpm_graph(matches, dice, roll)
    return True

def bipartite_match(G: list[list[bool]], u: int, seen: list[bool], matches: list[int]) -> bool:
    """
    Recurisively find a match for a node on a bipartite graph.

    Parameters
    ----------
    G : list of list of bool
        The adjacency matrix representing the graph.
    u : int
        The left node to try to match.
    seen : list of bool
        Markings for which right nodes have been seen so far.
    matches : list of int
        A list of which nodes have currently been matched.

    Returns
    -------
    True if `u` can be matched and False otherwise.
    """
    for v in range(len(matches)):
        if (G[u][v]) and not seen[v]:
            seen[v] = True
            if matches[v] < 0 or bipartite_match(G, matches[v], seen, matches):
                matches[v] = u
                return True
    return False

def find_all_possible_12_letter_words():
    """
    Find all the possible 12 letter words.
    """
    dice = load_dice(config.DICE_CSV_PATH)
    long_words = [word for word in load_words() if len(word) == 12 and is_possible_roll(list(word), dice)]
    print(long_words)