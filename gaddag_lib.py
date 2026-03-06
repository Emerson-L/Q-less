import utils
from treelib.tree import Tree
from collections import Counter
from typing import Optional
import pickle
from pathlib import Path

import config

def add_nodes(tree: Tree, parent_id: str, node: dict) -> None:
    """
    Recursively generates a tree to visualize a GADDAG.

    Parameters
    ----------
    tree : treelib.tree.Tree
        An empty tree to add to.
    parent_id : str
        The unique id of the parent node in the tree.
    node : dict
        The current node to add. The first node passed will be the
        root of the recursion.s
    """
    for letter, node in node.items():
        node_id = parent_id + letter
        
        tree.create_node(tag=letter, identifier=node_id, parent=parent_id)
        
        if isinstance(node, dict):
            add_nodes(tree, node_id, node)


def get_gaddag(words: list[str]) -> dict:
    """
    Generates a GADDAG from a provided lexicon.

    Parameters
    ----------
    words : list of str
        The lexicon to use.
    
    Returns
    -------
    dict
        The arising GADDAG. Each key is a letter or the delimiter `>`, while
        the values are more dicts that are the nodes themselves.
    """
    root = {}
    for w in words:
        for i in range(1, len(w)+1):
            cur_node = root
            if i == len(w):
                rep = w[::-1] + config.BREAK + config.EOW
            else:
                rep = w[0:i][::-1] + config.BREAK + w[i:len(w)] + config.EOW
            for l in rep:
                if l not in cur_node:
                    cur_node[l] = {}
                cur_node = cur_node[l]
    return root


def find_second_word(first_word: str, rack: Counter[str], gaddag: dict) -> Optional[tuple[int, list[tuple[int, str]]]]:
    """
    Docstring
    """
    for i, letter in enumerate(first_word):
        if letter not in gaddag:
            continue
        results = []
        _find_second_word(rack, gaddag[letter], letter, results)
        if results:
            return (i, list(map(get_word, results)))
    return

def _find_second_word(rack: Counter[str], node: dict, path: str, paths: list[str]) -> None:
    """
    Docstring
    """
    if config.EOW in node:
        return paths.append(path)
    for letter in rack:
        if letter not in node or rack[letter] == 0:
            continue
        rack[letter] -= 1
        _find_second_word(rack, node[letter], path + letter, paths)
        rack[letter] += 1
    if config.BREAK in node:
        _find_second_word(rack, node[config.BREAK], path + config.BREAK, paths)
    return

def get_word(path: str) -> tuple[int, str]:
    """
    Docstring
    """
    idx = path.find(config.BREAK) - 1
    prefix, suffix = path.split(config.BREAK)
    return (idx, prefix[::-1] + suffix)

def test_gaddag():
    """
    Docstring
    """
    words = utils.load_words(config.LEXICON_SOURCE)
    gaddag = get_gaddag(words)
    tree = Tree()
    tree.create_node(tag='Root', identifier='root')

    add_nodes(tree, 'root', gaddag)
    tree.show()

def load_gaddag():
    """
    Docstring
    """
    if Path(config.GADDAG_PKL_PATH).exists():
        with open(config.GADDAG_PKL_PATH, 'rb') as f:
            gaddag = pickle.load(f)
    else:
        words = utils.load_words(config.LEXICON_SOURCE)
        gaddag = get_gaddag(words)
        with open(config.GADDAG_PKL_PATH, 'wb') as f:
            pickle.dump(gaddag, f)
    return gaddag