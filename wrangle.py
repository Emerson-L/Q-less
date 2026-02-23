from emnist import extract_training_samples, extract_test_samples
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2 as cv

import utils
import config


def load_emnist_data(dataset_name:str) -> tuple[DataLoader, DataLoader]:
    """
    Loads emnist letters dataset into train and test DataLoaders

    Parameters
    ----------
    dataset_name : str
        name of emnist dataset to load (supports 'byclass' and 'letters')

    Returns
    -------
    train_loader : torch DataLoader
        DataLoader with train set
    test_loader : torch DataLoader
        DataLoader with test set
    """
    train_images, train_labels = extract_training_samples(dataset_name)
    test_images, test_labels = extract_test_samples(dataset_name)


    match dataset_name:
        case 'letters':
            train_labels = train_labels - 1
            test_labels = test_labels - 1
        case 'byclass':
            train_mask = (train_labels >= 10) & (train_labels <= 35)
            train_images = train_images[train_mask]
            train_labels = train_labels[train_mask] - 10
            test_mask = (test_labels >= 10) & (test_labels <= 35)
            test_images = test_images[test_mask]
            test_labels = test_labels[test_mask] - 10

    x_train = torch.tensor(train_images).unsqueeze(1).float() 
    y_train = torch.tensor(train_labels).long()

    x_test = torch.tensor(test_images).unsqueeze(1).float()
    y_test = torch.tensor(test_labels).long()

    x_train /= 255.0
    x_test /= 255.0

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return trainloader, testloader


def load_chars74k_data() -> tuple[DataLoader, DataLoader]:
    """
    Loads chars74k letters dataset into train and test DataLoaders

    Returns
    -------
    train_loader : torch DataLoader
        DataLoader with train set
    test_loader : torch DataLoader
        DataLoader with test set
    """

    images = []
    labels = []
    for character_dir in [p for p in Path(config.CHARS_74K_ENGLISH_PATH).glob('*') if p.is_dir()]:
        sample_num = int(character_dir.stem[-3:])
        character = utils.chars74k_sample_nums_to_characters([sample_num])[0]
        if character.isupper():
            for image_path in Path(character_dir).glob('*.png'):
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                resized = cv.resize(image, (28, 28), interpolation=cv.INTER_AREA)
                inverted = cv.bitwise_not(resized)
                image_tensor = torch.tensor(inverted).unsqueeze(0).float()
                image_tensor /= 255.0
                images.append(image_tensor)
                labels.append(utils.letters_to_numbers(character)[0])
        
    images = np.array(images)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
    x_train, x_test, y_train, y_test = torch.tensor(x_train), torch.tensor(x_test), torch.tensor(y_train), torch.tensor(y_test)
    
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return trainloader, testloader

def load_qless_test_data(test_images_dir:str) -> DataLoader:
    """
    Load Qless test data from a directory of 12 letter images and a labels.txt into a torch DataLoader

    Parameters
    ----------
    test_images_dir : str
        directory with .png images to load and a labels.txt with letter labels as a string on one line

    Returns
    -------
    testloader : torch DataLoader
        DataLoader with test dataset
    """
    images = np.zeros((config.NUM_DICE, 28, 28), dtype=np.uint8)
    for image_path in Path(test_images_dir).glob('*.png'):
        image_idx = int(image_path.stem.split('_')[1])
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        images[image_idx] = image

    with open(f'{test_images_dir}/labels.txt', 'r') as f:
        content = f.read()
        labels = utils.letters_to_numbers(content)

    x = torch.tensor(images).unsqueeze(1).float()
    y = torch.tensor(labels).long()

    x /= 255.0

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return loader