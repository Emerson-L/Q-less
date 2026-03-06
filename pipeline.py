import numpy as np
import cv2 as cv
from pathlib import Path

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
    true_dice = utils.load_dice(config.DICE_CSV_PATH)

    #TODO: Inefficient to load lexicon every time we solve, maybe just pass the lexicon to the class instantiation

    dice_image_dir = './assets/dice_images/paper_background/'
    for dice_image_path in Path(dice_image_dir).glob('*.JPG'):
        dice_image = cv.imread(dice_image_path)
        cv.imshow('Dice Image', dice_image)

        pred_letters = predict_letters_from_dice_image(dice_image_path, config.BENCHMARK_MODEL_PATH)
        pred_letters_lower = [char.lower() for char in pred_letters]

        print(f'Got roll: {sorted(pred_letters_lower)}')

        if not utils.is_possible_roll(pred_letters_lower, true_dice):
            msg = 'Roll is not possible, machine learning bad'
            print(msg)
            #raise ValueError(msg)

        solver = Solver(pred_letters_lower, verbose=True)
        solver.solve()

