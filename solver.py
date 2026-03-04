from dataclasses import dataclass
from typing import Optional
from collections import Counter

import numpy as np

import gaddag_lib
import utils
import config
import visualize

import logging

logging.basicConfig(
    filename='solver.log', 
    level=logging.INFO,
    filemode='w',
    format='%(message)s'
)

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
    def __init__(self, start_rack:list[str], verbose: bool = False):
        self.rack = Counter(start_rack)
        self.valid_words = utils.get_valid_words(start_rack, utils.load_words())
        self.gaddag = gaddag_lib.load_gaddag()
        self.board = self.make_empty_board()
        self.invalid_squares = set()
        self.verbose = verbose

    def make_empty_board(self):
        return np.full((config.BOARD_SIZE, config.BOARD_SIZE), '', dtype='<U1')
    
    def play_word(self, word:str, starting_pos:Coord):
        for idx, char in enumerate(word):
            self.board[(starting_pos + idx * HORIZONTAL).unpack()] = char.upper()

    def unplay_word(self, word:str, starting_pos:Coord, hook_pos:Optional[Coord]=None):
        for idx in range(len(word)):
            cur_pos = starting_pos + idx * HORIZONTAL
            if hook_pos and cur_pos == hook_pos:
                continue
            self.board[cur_pos.unpack()] = ''

    def solve(self) -> None:
        """
        Attempt to solve the Q-less game for a given roll.
        """
        for valid_word in sorted(self.valid_words, key=len, reverse=True):
            start_pos = Coord(7, 7 - len(valid_word) // 2)
            self.play_word(valid_word, start_pos)
            for letter in valid_word:
                self.rack[letter] -= 1
            self.board = self.board.T
            hooks = self.find_hooks()
            for hook in hooks:
                if self.dfs(self.gaddag, hook, '', -1):
                    return
            self.board = self.board.T
            self.unplay_word(valid_word, start_pos)
            for letter in valid_word:
                self.rack[letter] += 1

        visualize.plot_board(self.board, list(self.rack.elements()))

    
    def dfs(self, gaddag_node: dict, cur_pos: Coord, cur_word: str, direction: int) -> bool:
        """
        Recursively attempts to solve the Q-less board.

        Parameters
        ----------
        gaddag_node : dict
            The current position in the GADDAG based on the path so far.
        cur_pos : Coord
            The current position on the board based on the original start position.
        cur_word : str
            The raw path through the GADDAG so far, including the `BREAK` symbol and reversed prefixes.
        direction : int
            The direction for the implicit orientation of either 1 or -1. It is flipped when a `BREAK` edge is traversed.

        Returns
        -------
        bool
            `True` if the board can be solved and `False` otherwise. In the first case, the board will be in the solved state.
        """

        if self.verbose:
            logging.info(f"Calling dfs(self, \n\tgaddag_node={gaddag_node.keys()}, \n\tcur_pos={cur_pos}, \n\tcur_word={cur_word}, \n\tdirection={direction})")
            board_str = "Board:\n"
            for i, row in enumerate(self.board):
                for j, square in enumerate(row):
                    if (i, j) == cur_pos.unpack():
                        board_str += (square.lower() if square else '&') + ' '
                        continue
                    if (i, j) in self.invalid_squares:
                        board_str += "/ "
                        continue
                    board_str += (square if square else '#') + ' '
                board_str += '\n'
            logging.info(board_str)
            rack_str = "Rack:\n"
            for letter, count in self.rack.items():
                rack_str += f"{letter}: {count}\n"
            logging.info(rack_str)
            letter_str = "Letters committed:\n"
            for letter in cur_word:
                if letter != config.BREAK:
                    letter_str += letter + ' '
            logging.info(letter_str)
            logging.info(f"Direction: {">>" if direction == 1 else "<<"}")

        # Case 1: the rack is empty and all committed letters are played
        if self.empty_rack() and not cur_word:
            if self.verbose:
                logging.info("Done.")
                logging.info('-' * 50)
            return True
        
        # Case 2: the current square on the board is already filled and must be used
        square_symbol = str(self.board[cur_pos.unpack()]).lower()
        if self.verbose:
            logging.info(f"Current square: {square_symbol}\n")
        if square_symbol:
            if square_symbol in gaddag_node:
                if self.verbose:
                    logging.info(f"Trying letter {square_symbol}")
                    logging.info('-' * 50)
                return self.dfs(gaddag_node[square_symbol], cur_pos + HORIZONTAL * direction, cur_word + square_symbol, direction)
            if self.verbose:
                logging.info(f"Failed to use letter {square_symbol}")
            return False
        
        # Case 3: the current path in the GADDAG can be extended with a letter on the rack and the current position is legal to play in
        if cur_pos.unpack() not in self.invalid_squares:
            for letter in self.rack:
                if letter in gaddag_node and self.rack[letter]:
                    if self.verbose:
                        logging.info(f"Trying letter {letter}")
                        logging.info('-' * 50)
                    self.rack[letter] -= 1
                    if self.dfs(gaddag_node[letter], cur_pos + HORIZONTAL * direction, cur_word + letter, direction):
                        return True
                    if self.verbose:
                        logging.info(f"Failed with letter {letter}")
                    self.rack[letter] += 1

        # Case 4: it is possible to move from the prefix to the suffix
        if config.BREAK in gaddag_node:
            if self.verbose:
                logging.info(f"Trying BREAK")
                logging.info('-' * 50)
            if self.dfs(gaddag_node[config.BREAK], cur_pos + HORIZONTAL * (len(cur_word) + 1), cur_word + config.BREAK, -direction):
                return True
            if self.verbose:
                logging.info("Failed to use BREAK")
        
        # Case 5: The current GADDAG path can make a complete word which can be committed.
        if config.EOW in gaddag_node:
            idx, word = gaddag_lib.get_word(cur_word)
            start_pos = cur_pos - HORIZONTAL * len(word)
            self.play_word(word, start_pos)
            if self.verbose:
                logging.info(f"Playing word {word} at position {start_pos}")
                visualize.plot_board(self.board, list(self.rack.elements()))
            for _ in range(2):
                self.board = self.board.T
                hooks = self.find_hooks()
                for hook in hooks:
                    if self.dfs(self.gaddag, hook, '', -1):
                        return True
            if self.verbose:
                logging.info(f"Unplaying word {word}")
            self.unplay_word(word, start_pos, start_pos + idx * HORIZONTAL)
            self.find_hooks()
            return False
        
        return False


    def empty_rack(self) -> bool:
        """
        Check if the rack is empty or not.

        Returns
        -------
        bool
            `True` if the rack is empty, and `False` otherwise.
        """
        return all(count == 0 for count in self.rack.values())
        

    def is_valid_adjacent(self, square: Coord) -> bool:
        """
        Check if a square is valid to play in for the current orientation.

        Parameters
        ----------
        square : Coord
            The position of the square to check.

        Returns
        -------
        bool
            `True` if the square is possible and `False` otherwise.
        """
        # For now, only allow playing on an open board
        if not self.board[square.unpack()] and (self.board[(square + VERTICAL).unpack()] or self.board[(square - VERTICAL).unpack()]):
            return False
        return True
    
    def is_valid_hook(self, square: Coord) -> bool:
        if self.board[(square + HORIZONTAL).unpack()] + self.board[(square - HORIZONTAL).unpack()]:
            return False
        return True
    
    def coords_to_array(self, coords:list[tuple[int, int]]) -> Optional[np.ndarray]:
        coord_array = np.array(list(map(lambda x: np.array([x[0], x[1]]), coords)))
        if len(coord_array) == 0:
            return None
        return coord_array

    def find_hooks(self) -> list[Coord]:
        filled = list(map(lambda x: Coord(x[0], x[1]), np.argwhere(self.board != '')))
        hooks = list(filter(self.is_valid_hook, filled))
        self.update_board(filled)
        return hooks

    def update_board(self, filled:list[Coord]) -> None:
        self.invalid_squares = set()
        for coord in filled:
            for possible_square in [coord + HORIZONTAL, coord - HORIZONTAL]:
                if not self.is_valid_adjacent(possible_square):
                    self.invalid_squares.add(possible_square.unpack())
        
        highlight_arr = self.coords_to_array(list(self.invalid_squares))
        # visualize.plot_board(self.board, list(self.rack.elements()), highlight_arr)
    
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
