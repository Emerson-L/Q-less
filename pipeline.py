import numpy as np
import cv2 as cv
from pathlib import Path
import argparse

import find_dice
import modeling_torch
import utils
import config
from solver import Solver

def predict_letters_from_dice_image(image_path:str, model_path:str):
    """
    Takes in a .JPG image of 12 dice and makes predictions on the letters on the top of the dice

    Parameters
    ----------
    image_path : str
        path to a .JPG image of 12 dice to predict letters for
    model_path : str
        path to a .pth model to use for prediction
        
    Returns
    -------
    pred_letters : list of str
        list of predicted characters
    """
    letter_images = np.array(find_dice.generate_letter_images(image_path))
    preds, probs = modeling_torch.load_and_predict(letter_images, model_path)
    pred_letters = utils.numbers_to_letters(preds)

    return pred_letters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_path', type=str, help='Path to jpg/jpeg or dir of jpg/jpegs')
    args = parser.parse_args()

    images_path = Path(args.images_path)
    if images_path.is_dir():
        in_images = list(images_path.glob('*.[jJ][pP][gG]')) + list(images_path.glob('*.[jJ][pP][eE][gG]'))
    elif images_path.exists() and (images_path.match('*.[jJ][pP][gG]') or images_path.match('*.[jJ][pP][eE][gG]')):
        in_images = [images_path]
    else:
        raise ValueError(f'Invalid image directory or image path: {images_path}')

    true_dice = utils.load_dice(config.DICE_CSV_PATH)
    solver = Solver(verbose=True, show_final_board=True)

    for dice_image_path in in_images:
        print(f'Trying to solve {dice_image_path}')
        dice_image = cv.imread(dice_image_path)
        cv.imshow('Dice Image', dice_image)

        pred_letters = predict_letters_from_dice_image(dice_image_path, config.BENCHMARK_MODEL_PATH)
        pred_letters_lower = [char.lower() for char in pred_letters]

        print(f'Got roll: {sorted(pred_letters_lower)}')

        if not utils.is_possible_roll(pred_letters_lower, true_dice):
            msg = 'Roll is not possible, machine learning bad'
            print(msg)
            continue
            #raise ValueError(msg)

        solved = solver.solve(pred_letters_lower)

