
# CONSTANTS
NUM_DICE = 12
BOARD_SIZE = 15 # Assumes square board

# GADDAG CONSTANTS
BREAK = '>'
EOW = '*'

# MODELING
BATCH_SIZE = 32
NUM_EPOCHS = 10

# LEXICON CHOICES
LEXICON_SOURCE = 'nltk'              # Choose between 'twl' and 'nltk'
LEXICON_CORPUS_NLTK = 'wordnet31'   # Choose between nltk options

# FILE PATHS
DICE_CSV_PATH = './assets/dice.csv'
TRIE_PKL_PATH = './assets/trie.pkl'
GADDAG_PKL_PATH = './assets/gaddag.pkl'

CHARS_74K_ENGLISH_PATH = './assets/chars74k_English/Fnt/'
DICE_IMAGES_DIR = './assets/dice_images/paper_background/'
LETTER_IMAGES_DIR = './assets/letter_images/'

EMNIST_DATASET_NAME = 'byclass'
MODEL_PATH = f'./model_{NUM_EPOCHS}.pth'
BENCHMARK_MODEL_PATH = './benchmark_model_combined_10_aug_3random.pth'
