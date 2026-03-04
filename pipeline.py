import numpy as np

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
    pred_letters = predict_letters_from_dice_image('./assets/dice_images/paper_background/IMG_3296.JPG',
                                                   config.BENCHMARK_MODEL_PATH)
    pred_letters_lower = [char.lower() for char in pred_letters]
    s = Solver(pred_letters_lower)
    s.solve()
