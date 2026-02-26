import numpy as np
from enum import Enum
import random
from dataclasses import dataclass
from typing import Optional

import gaddag_lib
import utils
import config
import visualize

@dataclass(frozen=True)
class Coord:
    row: int
    col: int

    def __add__(self, other):
        """Allows you to add two Coord objects together using '+'"""
        if not isinstance(other, Coord):
            return NotImplemented
        return Coord(self.row + other.row, self.col + other.col)
    
    def __sub__(self, other):
        """Allows you to subtract two Coord objects using '-'"""
        if not isinstance(other, Coord):
            return NotImplemented
        return Coord(self.row - other.row, self.col - other.col)

    def __mul__(self, scalar: int):
        """Allows multiplying a direction Coord by an integer (steps)"""
        if not isinstance(scalar, int):
            return NotImplemented
        return Coord(self.row * scalar, self.col * scalar)
    
    def __rmul__(self, scalar: int):
        return self.__mul__(scalar)
    
    def __neg__(self):
        return Coord(-self.row, -self.col)

    def unpack(self) -> tuple:
        """Returns the tuple needed for NumPy indexing."""
        return (self.row, self.col)
    
VERTICAL = Coord(1, 0)
HORIZONTAL = Coord(0, 1)

class Solver:
    def __init__(self, start_rack:list[str]):
        self.rack = start_rack
        self.valid_words = utils.get_valid_words(start_rack, utils.load_words())
        self.gaddag = gaddag_lib.load_gaddag()
        self.board = self.make_empty_board()
        self.board_mask = self.make_board_mask()

    def make_empty_board(self):
        return np.full((13, 13), '', dtype='<U1')
    
    def make_board_mask(self):
        return np.full((13, 13), True, dtype=bool)
    
    def play_word(self, word:str, starting_pos:Coord, orientation:Coord):
        for idx, char in enumerate(word):
                self.board[(starting_pos + idx * orientation).unpack()] = char.upper()

    def unplay_word(self, word:str, starting_pos:Coord, orientation:Coord, hook_pos:Optional[Coord]=None):
        for idx, char in enumerate(word):
            cur_pos = starting_pos + idx * orientation
            if hook_pos and cur_pos == hook_pos:
                continue
            self.board[cur_pos.unpack()] = ''

    def solve(self):
        for valid_word in sorted(self.valid_words, key=len, reverse=True):
            start_pos = Coord(7, 7 - len(valid_word) // 2)
            self.play_word(valid_word, start_pos, HORIZONTAL)
            visualize.plot_board(self.board, self.rack)
            self.unplay_word(valid_word, start_pos, HORIZONTAL)
        # for each letter in that word, call dfs
    
    def dfs(self):
        pass

    def is_valid_square(self, square: Coord, orientation: Coord) -> bool:
        if len(self.board[(square + orientation).unpack()] + self.board[(square - orientation).unpack()]) == 1:
            return False
        return True

    def find_hooks(self):
        hooks = list(map(lambda x: Coord(x[0], x[1]), np.argwhere(self.board != '')))
        orientations = [VERTICAL, HORIZONTAL]
        for square in hooks:
            for sgn in [-1, 1]:
                for i in range(2):
                    major_axis, minor_axis = orientations[i], orientations[(i+1)%2]
                    adjacent = square + sgn * major_axis
                    if not self.board[adjacent.unpack()] and not self.is_valid_square(adjacent, minor_axis):
                        self.board_mask[adjacent] = False
                    else:
                        self.board_mask[adjacent] = True
        return hooks
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
