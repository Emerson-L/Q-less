import numpy as np
from pathlib import Path
import cv2 as cv
from skimage import measure, draw, transform
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

import visualize
import config

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

    #TODO: Contour value should be decided somehow based on the image, this is brittle but works for the test images 
    CONTOUR_VALUE = 60 # Value between 0-255 used to find contours in the grayscaled image

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
    for i in range(n_contours - config.NUM_DICE):
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

    Returns
    -------
    """
    PADDING_SCALAR = 1.3

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

        #visualize.plot_image(resized)

        dice_images.append(resized)

    return dice_images

def generate_letter_images(dice_image_path:str, letter_images_dir:str=None) -> list[np.ndarray]:
    """
    Generates 28x28 letter images from an image of dice

    Parameters
    ----------
    dice_image_path : str
        Path to a single .JPG image

    letter_images_dir : str
        path to a directory to put images in. Will be made and images written into it if provided.

    Returns
    -------
    out_images : list of np.ndarray
        List of 28x28 grayscaled images of letters
    """

    write = letter_images_dir is not None
    if write:
        Path(letter_images_dir).mkdir(exist_ok=True)

    image_path = Path(dice_image_path)
    
    out_images = []

    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    contours = get_contours(image)
    letter_images = contours_to_letter_images(image, contours)

    if write:
        this_image_dir = f'{letter_images_dir}/{image_path.stem}/'
        Path(this_image_dir).mkdir(exist_ok=True)
    
    for i, letter_image in enumerate(letter_images):
        cv2_image = (letter_image * 255).astype(np.uint8)
        cv2_inverted = cv.bitwise_not(cv2_image)
        out_images.append(cv2_inverted)

        if write:
            cv.imwrite(f'{this_image_dir}/letter_{i}.png', cv2_inverted)

    return out_images

if __name__ == '__main__':
    generate_letter_images(config.DICE_IMAGES_DIR, config.LETTER_IMAGES_DIR)





