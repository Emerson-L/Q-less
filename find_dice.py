import numpy as np
from pathlib import Path
import cv2 as cv
from skimage import measure, draw
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

import visualize

DICE_IMAGES_DIR = './assets/dice_images/paper_background/'
CONTOUR_VALUE = 50  # Value between 0-255 used to find contours in the grayscaled image 
NUM_DICE = 12

def get_contours(image:np.ndarray, n_contours:int = 20) -> None:
    """
    Finds contours within an image of the 12 letters on the top of the dice (usually) 
    First finds the top n_contours based on bounding box area, then pares that list down based on distance
    between dice and bounding box area

    Parameters
    ----------
    image: np.ndarray
        image loaded with cv.imread(image_path) and converted to grayscale
    n_contours : int
        number of contours to search through to find the correct contours that are on the top of the dice

    Returns
    -------
    best_num_dice_contours : list[np.ndarray]
        The chosen contours based on the number of dice
    """
    all_contours = measure.find_contours(image, CONTOUR_VALUE, fully_connected='high', positive_orientation='high')

    def get_contour_bounding_box_area(c):
        min_coords = np.min(c, axis=0)
        max_coords = np.max(c, axis=0)
        delta = max_coords - min_coords
        return delta[0] * delta[1]

    all_contours.sort(key=get_contour_bounding_box_area, reverse=True)
    largest_contours = all_contours[:n_contours]

    bbox_areas = [get_contour_bounding_box_area(c) for c in largest_contours]
    centers = [np.mean(c, axis = 0) for c in largest_contours]

    dist_matrix = distance_matrix(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)

    idxs_to_delete = []
    for i in range(n_contours - NUM_DICE):
        i = np.argmin(dist_matrix)
        i, j = np.unravel_index(i, dist_matrix.shape)

        idx_delete = i if bbox_areas[i] < bbox_areas[j] else j
        
        idxs_to_delete.append(idx_delete)
        dist_matrix[i, :] = np.inf
        dist_matrix[:, i] = np.inf
        dist_matrix[j, :] = np.inf
        dist_matrix[:, j] = np.inf

    best_num_dice_contours = [c for i, c in enumerate(largest_contours) if i not in idxs_to_delete]

    #visualize.plot_image_with_contours(image, best_n_contours)

    return best_num_dice_contours

def contours_to_letter_images(image:np.ndarray, contours:list[np.ndarray]) -> list[np.ndarray]:
    """
    Takes in contours and converts them to black/white 28x28 images

    Parameters
    ----------
    image : np.ndarray
        image that contains the given contours
    contours : list of np.ndarray
        list of contours to convert to images
    """

    for contour in contours:
        mask = np.zeros(image.shape, dtype=bool)
        rows, cols = draw.polygon(contour[:, 0], contour[:, 1], shape=image.shape)
        mask[rows, cols] = True

        extracted_pixels = np.zeros_like(image)
        extracted_pixels[mask] = image[mask]

        fig, ax = plt.subplots()
        ax.imshow(extracted_pixels, cmap=plt.cm.gray)
        plt.show()

        # still need to separate this into its own image based on bounding box and some padding
        # then threshold it decently and make it 28x28
        # add that to list of images

    return extracted_pixels

if __name__ == '__main__':
    for image_path in Path(DICE_IMAGES_DIR).glob('*.JPG'):
        image = cv.imread(image_path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  
        contours = get_contours(gray_image)
        letter_images = contours_to_letter_images(gray_image, contours)


