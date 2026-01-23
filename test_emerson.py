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
    return [random.choice(row).lower() for _, row in dice.iterrows()]

if __name__ == '__main__':
    letters = roll('./dice.csv')
    words = load_words()
    print(f'Rolled letters: {letters}')
    print(f'Total words: {len(words)}')

    letters_count = Counter(letters)
    valid_words = []
    for word in words:
        word_count = Counter(word)
        if word_count <= letters_count:
            valid_words.append(word)

    print(f'Found {len(valid_words)} valid starting words')
    lengths = [len(w) for w in valid_words]
    length_counts = Counter(lengths)
    for length in sorted(length_counts):
        print(f'Length {length:<10}  Count {length_counts[length]:<10}')


    