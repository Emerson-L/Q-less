from collections import Counter
import utils
import config
from tqdm import tqdm
import json

def get_unique_rolls(dice: list[list[str]]) -> dict[tuple[str], int]:
    """
    Calculates all unique rolls and their frequencies for a set of dice.

    Turns dice into generating polynomial functions, then multiplies them. Each term
    in the final product will be a roll and its coefficient will be the frequency.

    Parameters
    ----------
    dice : list of list of str
        The dice to find the combinations for.

    Returns
    -------
    dict
        Each unique roll as a tuple of characters and its frequency.
    """
    # Get polynomial for first die
    running_totals = Counter([(face,) for face in dice[0]])

    for die in tqdm(dice[1:], desc="Dice", position=0):
        # Get polynomial for next die
        die_counts = Counter([(face,) for face in die])

        # Perform multiplication
        new_totals = Counter()
        for roll_1, count_1 in tqdm(running_totals.items(), desc="Running total", position=1, leave=False):
            for roll_2, count_2 in die_counts.items():
                combined_roll = tuple(sorted(roll_1 + roll_2))
                new_totals[combined_roll] += count_1 * count_2
                
        running_totals = new_totals
        
    return running_totals

def main():
    dice = utils.load_dice(config.DICE_CSV_PATH)

    results = get_unique_rolls(dice)
    with open(config.COMBOS_JSON_PATH, 'w') as f:
        json.dump({''.join(key): val for key, val in results.items()}, f, separators=(',', ':'))
    print("Results saved to combos.json")

    print(f"Number of unique roll outcomes: {len(results)}")
    print(f"Total possible combinations calculated: {sum(results.values())}")

if __name__ == '__main__':
    main()