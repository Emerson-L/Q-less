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
    """
    A coordinate class to specify positions on `Solver.board`.

    Parameters
    ----------
    row : int
        The row of the board.
    col : int
        The column of the board.

    Attributes
    ----------
    row, col : see Parameters

    Notes
    -----
    This class supports addition of `Coord` instance as well as multiplication with scalars, allowing
    for easy positional updates. The initial values `row` and `col` should not be modified.
    """
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
        """
        Retrieve a tuple form of the coordinates.

        Returns
        -------
        tuple of int
            The first term is `row` and the second is `col`.
        """
        return (self.row, self.col)
    
    def transpose(self):
        """
        Get the transpose of the position.

        Returns
        -------
        Coord
            A new coordinated initialized with `row` and `col` reversed.
        """
        return Coord(self.col, self.row)
    
VERTICAL = Coord(1, 0)
HORIZONTAL = Coord(0, 1)

UPPER_CIRCLES = "ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ"

class Solver:
    """
    Object to solve a given Q-less roll.

    Parameters
    ----------
    start_rack : list of str
        The resulting letters from a roll.
    verbose : bool
        Flag for whether or not to log the solving prcess.
    show_plays : bool
        Flag for whether to visualize the solving process.

    Attributes
    ----------
    start_rack, verbose, visualize : see Parameters
    rack : Counter 
        The counts of all letters present in the roll.
    valid_words : list of str
        All possible words that can be made with the roll from the lexicon.
    gaddag : dict
        The pre-generated GADDAG data structure created from the lexicon.
    board : np.ndarray
        A 2D array to represent the Q-less board.
    invalid_squares : set of tuple of int
        A set of coordinates that can't be played on for a given board position.

    Methods
    -------
    solve()
        Attempt to solve the given roll.
    """
    def __init__(self, start_rack:list[str], verbose: bool = False, show_plays : bool = False):
        self.start_rack = start_rack
        self.rack = Counter(start_rack)
        self.valid_words = utils.get_valid_words(start_rack, utils.load_words())
        self.gaddag = gaddag_lib.load_gaddag()
        self.board = self.make_empty_board()
        self.invalid_squares = set()
        self.verbose = verbose
        self.show_plays = show_plays

    def make_empty_board(self) -> np.ndarray:
        """
        Initialize the Q-less board.

        Returns
        -------
        np.ndarray
            The board with size specified in `config.py`.
        """
        return np.full((config.BOARD_SIZE, config.BOARD_SIZE), '', dtype='<U1')
    
    def play_word(self, word:str, starting_pos:Coord) -> None:
        """
        Play a word onto the board.

        Parameters
        ----------
        word : str
            The word to play.
        starting_pos : Coord
            The position to start playing the word. Assumes horizontal orientation.
        """
        for idx, char in enumerate(word):
            self.board[(starting_pos + idx * HORIZONTAL).unpack()] = char.upper()

    def unplay_word(self, word:str, starting_pos:Coord, hook_pos:Optional[Coord]=None) -> None:
        """
        Remove a played word from the board.

        Parameters
        ----------
        word : str
            The word to remove.
        starting_pos : Coord
            The position to remove the word from.
        hook_pos : Coord, optional
            The position of a letter to keep if it was a hook.
        """
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
                    visualize.plot_board(self.board, self.start_rack)
                    return
            self.board = self.board.T
            self.unplay_word(valid_word, start_pos)
            for letter in valid_word:
                self.rack[letter] += 1

    
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

        # Logging block
        if self.verbose:
            logging.info('-' * 50)
            logging.info(f"Calling dfs(\n\tgaddag_node={gaddag_node.keys()}, \n\tcur_pos={cur_pos}, \n\tcur_word={cur_word}, \n\tdirection={direction}\n)")
            board_str = "Board:\n"
            for i, row in enumerate(self.board):
                for j, square in enumerate(row):
                    if (i, j) == cur_pos.unpack() or (i, j) in self.invalid_squares:
                        if (i, j) == cur_pos.unpack() and (i, j) in self.invalid_squares:
                            board_str += "🅮  "
                        elif (i, j) == cur_pos.unpack():
                            board_str += (UPPER_CIRCLES[ord(square) - 65] + ' ' if square else "█  ")
                        else:
                            board_str += "∅  "
                        continue
                    board_str += (square if square else '◌') + "  "
                board_str += '\n'
            logging.info(board_str)
            rack_str = "Rack: "
            for letter in self.rack.elements():
                rack_str += f"{letter.upper()} "
            logging.info(rack_str)
            letter_str = "Current path: "
            for letter in cur_word:
                letter_str += letter.upper() + ' '
            if not cur_word:
                letter_str += "None"
            logging.info(letter_str)
            logging.info(f"Direction: {'>>' if direction == 1 else '<<'}")

        # Case 1: the rack is empty and all committed letters are played
        if self.empty_rack() and not cur_word:
            if self.verbose:
                logging.info("Done.")
                logging.info('-' * 50)
            return True
        
        # Case 2: the current square on the board is already filled and must be used
        square_symbol = str(self.board[cur_pos.unpack()]).lower()
        if square_symbol:
            if square_symbol in gaddag_node:
                if self.verbose:
                    logging.info(f"Trying letter {square_symbol}.")
                    logging.info('-' * 50)
                return self.dfs(gaddag_node[square_symbol], cur_pos + HORIZONTAL * direction, cur_word + square_symbol, direction)
            if self.verbose:
                logging.info(f"Failed to use letter {square_symbol}.")
                logging.info('-' * 50)
            return False
        
        # Case 3: the current path in the GADDAG can be extended with a letter on the rack and the current position is legal to play in
        if cur_pos.unpack() not in self.invalid_squares:
            for letter in self.rack:
                if letter in gaddag_node and self.rack[letter]:
                    if self.verbose:
                        logging.info(f"Trying letter {letter}.")
                    self.rack[letter] -= 1
                    if self.dfs(gaddag_node[letter], cur_pos + HORIZONTAL * direction, cur_word + letter, direction):
                        logging.info('-' * 50)
                        return True
                    if self.verbose:
                        logging.info(f"Failed with letter {letter}.")
                    self.rack[letter] += 1

        # Case 4: it is possible to move from the prefix to the suffix
        if config.BREAK in gaddag_node:
            if self.verbose:
                logging.info("Trying BREAK.")
            if self.dfs(gaddag_node[config.BREAK], cur_pos + HORIZONTAL * (len(cur_word) + 1), cur_word + config.BREAK, -direction):
                logging.info('-' * 50)
                return True
            if self.verbose:
                logging.info("Failed to use BREAK.")
        
        # Case 5: The current GADDAG path can make a complete word which can be committed.
        if config.EOW in gaddag_node:
            idx, word = gaddag_lib.get_word(cur_word)
            start_pos = cur_pos - HORIZONTAL * len(word)
            self.play_word(word, start_pos)
            if self.verbose:
                logging.info(f"Playing word {word} at position {start_pos}.")
            if self.show_plays:
                visualize.plot_board(self.board, list(self.rack.elements()))
            for _ in range(2):
                self.board = self.board.T
                hooks = self.find_hooks()
                for hook in hooks:
                    if self.dfs(self.gaddag, hook, '', -1):
                        logging.info('-' * 50)
                        return True
            if self.verbose:
                logging.info(f"Unplaying word {word}.")
                logging.info('-' * 50)
            self.unplay_word(word, start_pos, start_pos + idx * HORIZONTAL)
            self.find_hooks()
            return False
        logging.info('-' * 50)
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

        Notes
        -----
        For now, a square is said to be a valid adjacent if there are no letters next to it along the axis
        that isn't being played along. This is far from a complete picture, as it may be possible to play
        in a square if a valid word is simultaneously formed along the other axis.
        """
        # For now, only allow playing on an open board
        if not self.board[square.unpack()] and (self.board[(square + VERTICAL).unpack()] or self.board[(square - VERTICAL).unpack()]):
            return False
        return True
    
    def is_valid_hook(self, square: Coord) -> bool:
        """
        Check if a placed letter is a possible hook or not.

        Parameters
        ----------
        square : Coord
            The position of the square to check.

        Returns
        -------
        bool
            `True` if the square is a possible hook and `False` otherwise.

        Notes
        -----
        Technically, any filled square is a valid hook. However, this function filters out hooks that
        would extend existing words, as longer words are guaranteed to have been already played.
        """
        if self.board[(square + HORIZONTAL).unpack()] + self.board[(square - HORIZONTAL).unpack()]:
            return False
        return True
    
    def coords_to_array(self, coords:list[tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Convert a list of coordinates to a NumPy array format.

        Parameters
        ----------
        coords : list of tuple of int
            Raw tuples representing coordinates on the board.

        Returns
        -------
        np.ndarray
            A converted 2d array that is compatible with indexing operations, or None if coords is empty.
        """
        coord_array = np.array(list(map(lambda x: np.array([x[0], x[1]]), coords)))
        if len(coord_array) == 0:
            return None
        return coord_array

    # TODO : Refactor these methods so they aren't so weirdly connected. It seems that either
    # get_filled_squares() should be a separate method, or filled_squares should be a stored
    # and updated class attribute.
    def find_hooks(self) -> list[Coord]:
        """
        Find all hooks for a given board state and update the board.

        Returns
        -------
        list of Coord
            The positions on the board for each valid hook.
        """
        filled = list(map(lambda x: Coord(x[0], x[1]), np.argwhere(self.board != '')))
        hooks = list(filter(self.is_valid_hook, filled))
        self.update_board(filled)
        return hooks

    def update_board(self, filled:list[Coord]) -> None:
        """
        Update which squares are valid to play on for the given board state.

        Parameters
        ----------
        filled : list of Coord
            The positions of all filled squares on the board.
        """
        self.invalid_squares = set()
        for coord in filled:
            for possible_square in [coord + HORIZONTAL, coord - HORIZONTAL]:
                if not self.is_valid_adjacent(possible_square):
                    self.invalid_squares.add(possible_square.unpack())



if __name__ == '__main__':
    dice = utils.load_dice(config.DICE_CSV_PATH)
    letters = utils.roll(dice)
    solver = Solver(letters, verbose=True)
    solver.solve()
