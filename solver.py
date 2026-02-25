import numpy as np
from enum import Enum
import random

import gaddag_lib
import utils
import config
import visualize

class Orientation(Enum):
    VERTICAL = 1
    HORIZONTAL = 2

class Solver:
    def __init__(self, start_rack:list[str]):
        self.rack = start_rack
        self.valid_words = utils.get_valid_words(letters, utils.load_words())
        self.gaddag = gaddag_lib.load_gaddag()
        self.board = self.make_empty_board()

    def make_empty_board(self):
        return np.full((13, 13), '', dtype='<U1')
    
    def play_word(self, word:str, starting_pos:tuple[int, int], orientation:Orientation):
        for idx, char in enumerate(word):
            if orientation == Orientation.HORIZONTAL:
                self.board[starting_pos[0], starting_pos[1] + idx] = char.upper()
            if orientation == Orientation.VERTICAL:
                self.board[starting_pos[0] + idx, starting_pos[1]] = char.upper()

    def unplay_word(self, word:str, starting_pos:tuple[int, int], hook_pos:tuple[int, int], orientation:Orientation):
        for idx, char in enumerate(word):
            # row = 
            # col = 
            # if 
            if orientation == Orientation.HORIZONTAL:
                self.board[starting_pos[0], starting_pos[1] + idx] = ''
            if orientation == Orientation.VERTICAL:
                self.board[starting_pos[0] + idx, starting_pos[1]] = ''
        # if pos = hook_pos we continue

    def solve(self):
        for valid_word in sorted(self.valid_words, key=len, reverse=True):
            self.play_word(valid_word, (7, 7 - len(valid_word) // 2), Orientation.HORIZONTAL)
            visualize.plot_board(solver.board, letters)
        # for each letter in that word, call dfs
    
    def dfs(self):
        pass

    def find_hooks():
        pass
        # find_adjacents
        # then find_adjacents of those adjacents
        # has no adjacents thats fine
        # has two adjacents thats fine



if __name__ == '__main__':
    dice = utils.load_dice(config.DICE_CSV_PATH)
    letters = utils.roll(dice)
    solver = Solver(letters)
    #solver.play_word('blarf', (11, 11), Orientation.HORIZONTAL)
    solver.solve()
