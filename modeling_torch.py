"""
modeling_torch.py
For training, testing, and saving a model using pytorch
Derived from pytorch CNN tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Example usage to train model: python modeling_torch.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import visualize
import utils
import config
import wrangle

class Net(torch.nn.Module):
    """
    CNN architecture derived from pytorch CNN tutorial
    """
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(2, 16, 5)
        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(trainloader:DataLoader, model_path:str) -> np.ndarray:
    """
    Trains and saves model. Derived from pytorch CNN tutorial

    Parameters
    ----------
    trainloader : torch DataLoader
        DataLoader with training dataset
    model_path : str
        path to save trained model to

    Returns
    -------
    losses : np.ndarray
        1d array of training losses
    """
    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    losses = []
    for epoch in range(config.NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                losses.append(running_loss)
                running_loss = 0.0

    torch.save(net.state_dict(), model_path)
    print('Finished training, saved model')
    return np.array(losses)


def load_and_test(testloader:DataLoader, model_path:str, plot_wrong_predictions:bool=False) -> float:
    """
    Loads and tests a given model on a given test set. Derived from pytorch CNN tutorial

    Parameters
    ---------
    testloader : torch DataLoader
        DataLoader with test set
    model_path : str
        path to existing saved model
    
    Returns
    -------
    accuracy : float
        accuracy on test set
    """
    net = Net()
    net.load_state_dict(torch.load(model_path, weights_only=True))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            softmaxed_outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if plot_wrong_predictions:
                for i, (image, pred, label) in enumerate(zip(images, predicted, labels)):
                    if pred != label:
                        letter = utils.numbers_to_letters([pred])[0]
                        visualize.plot_image(image[0], letter=letter)
                        visualize.plot_probs(utils.numbers_to_letters(range(26)), softmaxed_outputs[i])

    accuracy = 100 * correct / total
    return accuracy

def load_and_predict(letter_images:list[np.ndarray], model_path:str) -> tuple[list[int], list[float]]:
    """
    Loads unlabeled images and make predictions on them

    Parameters
    ----------
    letter_images : list of np.ndarray
        list of images
    model_path : str
        path to model .pth to use for prediction
    
    Returns
    -------
    predicted : list of int
        list of predictions for each image
    softmaxed_outputs : list of float
        lists of probabilities for each class (A-Z) for each image
    """
    images = torch.tensor(letter_images).unsqueeze(1).float()
    images /= 255.0

    net = Net()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    outputs = net(images)
    softmaxed_outputs = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
   
    predicted = predicted.tolist()
    softmaxed_outputs = softmaxed_outputs.tolist()

    return predicted, softmaxed_outputs

def eval_labeled_qless_test_data(model_path:str, plot_wrong_predictions:bool=False):
    """
    Loads a given model and tests on all Qless data that has been labeled

    Parameters
    ----------
    model_path : str
        path to a model.pth to load and test
    plot_wrong_predictions : bool
        whether to plot the predictions that weren't correct

    Returns
    -------
    avg_accuracy : float
        average accuracy across all letters
    """
    
    test_images_dirs = [dir for dir in Path('./assets/letter_images').glob('*') if dir.is_dir()]

    Qless_accuracies = []
    for dir in test_images_dirs:
        if Path(f'{dir}/labels.txt').exists():
            Qless_testloader = wrangle.load_qless_test_data(dir)
            Qless_accuracy = load_and_test(Qless_testloader, model_path, plot_wrong_predictions=plot_wrong_predictions)
            print(f'Accuracy on {Path(dir).stem}: {Qless_accuracy:.2f}%')
            Qless_accuracies.append(Qless_accuracy)
    avg_accuracy = np.mean(Qless_accuracies)
    print(f'Average accuracy across dirs: {avg_accuracy:.2f}%')
    return avg_accuracy

if __name__ == '__main__':

    n_augments_rotation = 3

    # Train and test EMNIST
    #x_train, y_train, x_test, y_test = wrangle.load_emnist_data(config.EMNIST_DATASET_NAME, n_augments_rotation=n_augments_rotation)
    
    # Train and test chars74k
    #x_train, y_train, x_test, y_test = wrangle.load_chars74k_data(n_augments_rotation=n_augments_rotation)

    # Use a combination of the two datasets instead
    x_train_emnist, y_train_emnist, x_test_emnist, y_test_emnist = wrangle.load_emnist_data(config.EMNIST_DATASET_NAME, n_augments_rotation=n_augments_rotation)
    x_train_chars, y_train_chars, x_test_chars, y_test_chars = wrangle.load_chars74k_data(n_augments_rotation=n_augments_rotation)
    x_train = torch.cat((x_train_emnist, x_train_chars))
    y_train = torch.cat((y_train_emnist, y_train_chars))
    x_test = torch.cat((x_test_emnist, x_test_chars))
    y_test = torch.cat((y_test_emnist, y_test_chars))

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    trainloader, testloader = wrangle.make_dataloaders(x_train, y_train, x_test, y_test)

    losses = train(trainloader, config.MODEL_PATH)
    visualize.plot_loss_curve(losses)

    accuracy = load_and_test(testloader, config.MODEL_PATH)
    print(f'Accuracy on test set: {accuracy:.2f}%')

    #eval_labeled_qless_test_data('./model_combined_10_aug_3random.pth')

