import numpy as np
import matplotlib.pyplot as plt

def plot_board(board:np.ndarray, letters:list[str], adjacents:np.ndarray=None, output_file:str=None):
    """
    Plots an image of the given board and either saves it to output_file if given or shows the plot.

    Parameters
    ----------
    board : 2d numpy array of str
        The Q-less game board
    letters : list of str
        The rolled letters for displaying at the bottom of the plot
    adjacents : numpy array of (int, int)
        Optional coordinates of squares that are adjacent to already filled sqaures to display
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
        '   '.join(l.upper() for l in letters),
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=14,
        fontweight='bold'
    )

    # Plot letters
    for i in range(width):
        for j in range(width):
            ax.text(
                j, i, board[i, j],
                ha='center', va='center',
                fontsize=12, fontweight='bold'
            )

    # Color the adjacents
    mask = np.zeros((24, 24))
    mask[adjacents[:, 0], adjacents[:, 1]] = 1
    ax.imshow(
        mask,
        cmap='Reds',
        alpha=0.5,
        vmin=0,
        vmax=1
    )

    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

