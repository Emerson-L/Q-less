import pandas as pd
import random
import nltk
from collections import Counter

def load_words():
    """
    Loads nltk words that are valid for Q-less
    """
    nltk.download('words')
    from nltk.corpus import words
    return [w for w in words.words() if w.islower() and len(w) >= 3 and w.isalpha()]


def roll(dice_csv:str):
    """
    Picks 12 random letters from the Q-less dice
    """
    dice = pd.read_csv(dice_csv, header=None)
    return [random.choice(row) for _, row in dice.iterrows()]

if __name__ == '__main__':
    letters = roll('./dice.csv')
    words = load_words()
    print(letters)
    print(len(words))

    valid_words = []
    for word in words:
        word_count = Counter(word)
        print(word_count)

    