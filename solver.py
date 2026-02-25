import networkx as nx
import matplotlib.pyplot as plt

from gaddag_lib import get_gaddag
import utils

DICE_CSV_PATH = './assets/dice.csv'
TRIE_PKL_PATH = './assets/trie.pkl'
GADDAG_PKL_PATH = './assets/gaddag.pkl'


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
    # visualize_graph(matches, dice, roll)
    return True

def bpm(G: list[list[bool]], u: int, seen: list[bool], matches: list[int]) -> bool:
    for v in range(len(matches)):
        if (G[u][v]) and not seen[v]:
            seen[v] = True
            if matches[v] < 0 or bpm(G, matches[v], seen, matches):
                matches[v] = u
                return True
    return False

def visualize_graph(matches: list[int], dice: list[list[str]], roll: list[str]) -> None:
    dice_nodes = list(map(lambda x: ''.join(x).upper(), dice))
    B = nx.Graph()

    # Add nodes with a 'bipartite' attribute to identify their set
    B.add_nodes_from(roll, bipartite=0)
    B.add_nodes_from(dice_nodes, bipartite=1)

    edges = []
    for i in range(len(matches)):
        edges.append((roll[matches[i]], dice_nodes[i]))

    B.add_edges_from(edges)

    pos = nx.bipartite_layout(B, roll)

    # 5. Draw the graph
    plt.figure(figsize=(8, 5)) # Optional: adjust figure size
    nx.draw_networkx(
        B,
        pos=pos,
        with_labels=True,      # Show node labels
        node_color=[
            'skyblue' if node in roll else 'lightgreen' for node in B.nodes
        ],
        node_size=700,         # Adjust node size
        edge_color='gray',     # Adjust edge color
        font_size=12           # Adjust label font size
    )
    plt.title("Bipartite Graph Visualization")
    plt.axis('off') # Hide axis
    plt.show()

def main():
    dice = utils.load_dice(DICE_CSV_PATH)
    long_words = [word for word in utils.load_words() if len(word) == 12 and is_possible_roll(list(word), dice)]

if __name__ == '__main__':
    main()
