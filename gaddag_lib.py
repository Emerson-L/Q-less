import utils
from treelib.tree import Tree
import main
from collections import Counter
from typing import Optional
import random

BREAK = '>'
EOW = '*'

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
                rep = w[::-1] + BREAK + EOW
            else:
                rep = w[0:i][::-1] + BREAK + w[i:len(w)] + EOW
            for l in rep:
                if l not in cur_node:
                    cur_node[l] = {}
                cur_node = cur_node[l]
    return root


def find_second_word(first_word: str, rack: Counter[str], gaddag: dict) -> Optional[tuple[int, list[tuple[int, str]]]]:
    for i, letter in enumerate(first_word):
        if letter not in gaddag:
            continue
        results = []
        _find_second_word(rack, gaddag[letter], letter, results)
        if results:
            return (i, list(map(get_word, results)))
    return

def _find_second_word(rack: Counter[str], node: dict, path: str, paths: list[str]) -> None:
    if EOW in node:
        return paths.append(path)
    for letter in rack:
        if letter not in node or rack[letter] == 0:
            continue
        rack[letter] -= 1
        _find_second_word(rack, node[letter], path + letter, paths)
        rack[letter] += 1
    if BREAK in node:
        _find_second_word(rack, node[BREAK], path + BREAK, paths)
    return

def get_word(path: str) -> tuple[int, str]:
    idx = path.find(BREAK) - 1
    prefix, suffix = path.split(BREAK)
    return (idx, prefix[::-1] + suffix)




if __name__ == '__main__':
    dice = utils.load_dice(main.DICE_CSV_PATH)
    letters = utils.roll(dice)
    rack = Counter(letters)
    words = utils.load_words()
    gaddag = get_gaddag(words)
    valid_words = utils.get_valid_words(letters, words)
    starter = random.choice(valid_words)
    print(f"Original Rack: {' '.join(letters).upper()}")
    for letter in starter:
        rack[letter] -= 1
        letters.remove(letter)
    results = find_second_word(starter, rack, gaddag)
    if results:
        i, (j, word) = results[0], results[1][0]
    print(f"First word: {starter}")
    print(f"New rack: {' '.join(letters).upper()}")
    print(f"Second word: {word}")
    print(f"Hook: {starter[i]} (index {i})")

    # tree = Tree()
    # tree.create_node(tag='Root', identifier='root')

    # add_nodes(tree, 'root', gaddag)
    # tree.show()