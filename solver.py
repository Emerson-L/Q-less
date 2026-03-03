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
    
    def transpose(self):
        return Coord(self.col, self.row)
    
VERTICAL = Coord(1, 0)
HORIZONTAL = Coord(0, 1)

class Solver:
    def __init__(self, start_rack:list[str]):
        self.rack = start_rack
        self.valid_words = utils.get_valid_words(start_rack, utils.load_words())
        self.gaddag = gaddag_lib.load_gaddag()
        self.board = self.make_empty_board()
        self.invalid_squares = set()

    def make_empty_board(self):
        return np.full((config.BOARD_SIZE, config.BOARD_SIZE), '', dtype='<U1')
    
    def play_word(self, word:str, starting_pos:Coord):
        for idx, char in enumerate(word):
            self.board[(starting_pos + idx * HORIZONTAL).unpack()] = char.upper()

    def unplay_word(self, word:str, starting_pos:Coord, hook_pos:Optional[Coord]=None):
        for idx, char in enumerate(word):
            cur_pos = starting_pos + idx * HORIZONTAL
            if hook_pos and cur_pos == hook_pos:
                continue
            self.board[cur_pos.unpack()] = ''

    def solve(self):
        for valid_word in sorted(self.valid_words, key=len, reverse=True):
            start_pos = Coord(7, 7 - len(valid_word) // 2)

            self.play_word(valid_word, start_pos)
            self.find_hooks()
            self.unplay_word(valid_word, start_pos)

    
    def dfs(self):
        pass

    def is_valid_adjacent(self, square: Coord) -> bool:
        if len(self.board[(square + VERTICAL).unpack()] + self.board[(square - VERTICAL).unpack()]) == 1:
            return False
        return True
    
    def is_valid_hook(self, square: Coord) -> bool:
        if len(self.board[(square + HORIZONTAL).unpack()] + self.board[(square - HORIZONTAL).unpack()]) > 0:
            return False
        return True
    
    def coords_to_array(self, coords:list[Coord]) -> np.ndarray:
        coord_array = np.array(list(map(lambda x: np.array([x.row, x.col]), coords)))
        if len(coord_array) == 0:
            return None
        return coord_array

    def find_hooks(self):
        filled = list(map(lambda x: Coord(x[0], x[1]), np.argwhere(self.board != '')))
        hooks = list(filter(self.is_valid_hook, filled))
        self.update_board(filled)
        return hooks

    def update_board(self, filled:list[Coord]):
        self.invalid_squares = set()
        for coord in filled:
            for possible_square in [coord + HORIZONTAL, coord - HORIZONTAL]:
                if not self.is_valid_adjacent(possible_square):
                    self.invalid_squares.add(possible_square)
        
        print(self.board)
        highlight_arr = self.coords_to_array(list(self.invalid_squares))
        visualize.plot_board(self.board, self.rack, highlight_arr)
    
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
    solver.find_hooks()
