import matplotlib.pyplot as plt

from gaddag_lib import get_gaddag
import utils
import config


def is_possible_roll(roll: list[str], dice: list[list[str]]) -> bool:
    if len(roll) != len(dice):
        print("Roll length and dice length must match.")
        return False
    dice_sides = list(map(set, dice))
    G = [[False for _ in range(len(dice))] for _ in range(len(dice))]
    for i, letter in enumerate(roll):
        for j, sides in enumerate(dice_sides):
            if letter in sides:
                G[i][j] = True
    matches = [-1] * len(dice)
    for u in range(len(roll)):
        seen = [False] * len(dice)
        if not bpm(G, u, seen, matches):
            return False
    # visualize_bpm_graph(matches, dice, roll)
    return True

def bpm(G: list[list[bool]], u: int, seen: list[bool], matches: list[int]) -> bool:
    for v in range(len(matches)):
        if (G[u][v]) and not seen[v]:
            seen[v] = True
            if matches[v] < 0 or bpm(G, matches[v], seen, matches):
                matches[v] = u
                return True
    return False

def main():
    dice = utils.load_dice(config.DICE_CSV_PATH)
    long_words = [word for word in utils.load_words() if len(word) == 12 and is_possible_roll(list(word), dice)]

if __name__ == '__main__':
    main()
