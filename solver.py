from gaddag_lib import get_gaddag
import utils

DICE_CSV_PATH = './assets/dice.csv'
TRIE_PKL_PATH = './assets/trie.pkl'
GADDAG_PKL_PATH = './assets/gaddag.pkl'

def main():
    dice = utils.load_dice(DICE_CSV_PATH)
    letters = utils.roll(dice)
    words = utils.load_words()
    print(f'Rolled letters: {letters}')
    valid_words = utils.get_valid_words(letters, words)

if __name__ == '__main__':
    pass
