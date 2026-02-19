import numpy as np
from pathlib import Path
import cv2 as cv
from skimage import measure, draw, transform
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

import visualize

DICE_IMAGES_DIR = './assets/dice_images/paper_background/'
NUM_DICE = 12

CONTOUR_VALUE = 50  # Value between 0-255 used to find contours in the grayscaled image
#TODO: Contour value should be decided somehow based on the image, this is brittle but works for the test images 
PADDING_SCALAR = 1.3

def get_contours(image:np.ndarray, n_contours:int = 20) -> list[np.ndarray]:
    """
    Finds contours within an image of the 12 letters on the top of the dice (usually) 
    First finds the top n_contours based on bounding box area, then pares that list down based on distance
    between dice and bounding box area

    Parameters
    ----------
    image: np.ndarray
        Image array in grayscale
    n_contours : int
        Number of contours to search through to find the correct contours that are on the top of the dice

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

    #visualize.plot_image_with_contours(image, best_num_dice_contours)

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

    dice_images = []
    for contour in contours:
        contour_mask = np.zeros(image.shape, dtype=bool)
        rows, cols = draw.polygon(contour[:, 0], contour[:, 1], shape=image.shape)
        contour_mask[rows, cols] = True

        contoured_pixels = np.ones_like(image) * 255
        contoured_pixels[contour_mask] = image[contour_mask]

        r_min, c_min = np.min(contour, axis=0)
        r_max, c_max = np.max(contour, axis=0)
        height = r_max - r_min
        width = c_max - c_min
        center_r, center_c = (r_min + r_max) / 2, (c_min + c_max) / 2
        side_length = max(height, width) * PADDING_SCALAR

        r_min, r_max = max(0, int(center_r - side_length/2)), min(image.shape[0], int(center_r + side_length/2))
        c_min, c_max = max(0, int(center_c - side_length/2)), min(image.shape[1], int(center_c + side_length/2))

        cropped = contoured_pixels[r_min:r_max, c_min:c_max]

        resized = transform.resize(cropped, (28, 28), anti_aliasing=True)

        visualize.plot_image(resized)

        dice_images.append(resized)

    return dice_images

if __name__ == '__main__':
    for image_path in Path(DICE_IMAGES_DIR).glob('*.JPG'):
        image = cv.imread(image_path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  
        contours = get_contours(gray_image)
        letter_images = contours_to_letter_images(gray_image, contours)





