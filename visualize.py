import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional

import config

def plot_board(board:np.ndarray, letters:list[str], highlights:Optional[np.ndarray]=None, output_file:Optional[str]=None):
    """
    Plots an image of the given board and either saves it to output_file if given or shows the plot.

    Parameters
    ----------
    board : 2d numpy array of str
        The Q-less game board
    letters : list of str
        The rolled letters for displaying at the bottom of the plot
    highlights : numpy array of (int, int)
        Optional coordinates of squares that are to highlight
    output_file : str
        File path ending in .png to optionally save the board image
    """

    width = board.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.tight_layout(pad=1)

    ax.imshow(np.zeros((width, width)), cmap='gray_r')

    ax.set_xticks(np.arange(-0.5, width, 1))
    ax.set_yticks(np.arange(-0.5, width, 1))
    ax.grid(color='black', linewidth=0.4)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    ax.text(
        0.5, -0.02,
        '   '.join(letter.upper() for letter in sorted(letters)),
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=14,
        fontweight='bold'
    )

    for i in range(width):
        for j in range(width):
            ax.text(j, i, board[i, j], ha='center', va='center', fontsize=12, fontweight='bold')

    if highlights is not None:
        mask = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE))
        mask[highlights[:, 0], highlights[:, 1]] = 1
        ax.imshow(mask, cmap='Reds', alpha=0.5, vmin=0,vmax=1)

    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_image_with_contours(image:np.ndarray, contours:list[np.ndarray], output_file:Optional[str]=None) -> None:
    """
    Plot the given image with all given contours highlighted

    Parameters
    ----------
    image : np.ndarray
        Image array
    contours : list of np.ndarray
        list of contours from sklearn's measure.find_contours()
    output_file : str
        File path ending in .png to optionally save the board image
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_image(image:np.ndarray, letter:str=None, output_file:Optional[str]=None) -> None:
    """
    Plot the given image in grayscale

    Parameters
    ----------
    image : np.ndarray
        Image array
    letter : character
        letter to plot in the top left, i.e. the predicted letter
    output_file : str
        File path ending in .png to optionally save the board image
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    if letter is not None:
        ax.text(0.5, 2, letter, fontsize=20, color='white')
    
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_probs(letters:list[str], probs:np.ndarray, output_file:Optional[str]=None) -> None:
    """
    Plots probabilities of each letter given letters and 

    Parameters
    ----------
    letters : list of str
        List of letters in alphabet
    probs : 1d np.array of float
        Array of probabilites corresponding to each letter
    output_file : str
        File path ending in .png to optionally save the board image
    """
    plt.bar(letters, probs)
    plt.ylim(0, 1)
    plt.xlabel('Letter')
    plt.ylabel('Probability')
   
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_loss_curve(losses:np.ndarray, output_file:Optional[str]=None) -> None:
    """
    Plots loss curve

    Parameters
    ----------
    losses : np.ndarray
        Array of loss values from training
    output_file : str
        File path ending in .png to optionally save the board image
    """
    plt.plot(losses)
    plt.xlabel('Epoch/Batch')
    plt.ylabel('Loss')
   
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_bpm_graph(matches: list[int], dice: list[list[str]], roll: list[str]) -> None:
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
