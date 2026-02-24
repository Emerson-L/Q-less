import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import cv2 as cv

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
        # print("1")
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print("2")
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print("3")
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print("4")
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print("5")
        # print(x.shape)
        x = self.fc3(x)
        # print("6")
        # print(x.shape)
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

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

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

if __name__ == '__main__':

    model_to_test = './model_byclass_10_aug_3random.pth'
    n_augments_rotation = 3

    # Train and test EMNIST
    #x_train, y_train, x_test, y_test = wrangle.load_emnist_data(config.EMNIST_DATASET_NAME, n_augments_rotation=n_augments_rotation)
    
    # Train and test chars74k
    #x_train, y_train, x_test, y_test = wrangle.load_chars74k_data(n_augments_rotation=n_augments_rotation)

    # Use a combination of the two datasets instead
    x_train_emnist, y_train_emnist, x_test_emnist, y_test_emnist = wrangle.load_emnist_data(config.EMNIST_DATASET_NAME, n_augments_rotation=n_augments_rotation)
    x_train_chars, y_train_chars, x_test_chars, y_test_chars = wrangle.load_chars74k_data(n_augments_rotation=n_augments_rotation)
    x_train = np.concatenate((x_train_emnist, x_train_chars))
    y_train = np.concatenate((y_train_emnist, y_train_chars))
    x_test = np.concatenate((x_test_emnist, x_test_chars))
    y_test = np.concatenate((y_test_emnist, y_test_chars))

    trainloader, testloader = wrangle.make_dataloaders(x_train, y_train, x_test, y_test)

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    losses = train(trainloader, config.MODEL_PATH)
    visualize.plot_loss_curve(losses)

    accuracy = load_and_test(testloader, model_to_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')

    # Test our model on real Qless data
    test_images_dirs = ['./assets/letter_images/IMG_3296/',
                         './assets/letter_images/IMG_3297/',
                         './assets/letter_images/IMG_3298/',
                         './assets/letter_images/IMG_3299/',]
    
    Qless_accuracies = []
    for dir in test_images_dirs:
        Qless_testloader = wrangle.load_qless_test_data(dir)
        Qless_accuracy = load_and_test(Qless_testloader, model_to_test, plot_wrong_predictions=True)
        print(f'Accuracy on {Path(dir).stem}: {Qless_accuracy:.2f}%')
        Qless_accuracies.append(Qless_accuracy)
    print(f'Average accuracy across dirs: {np.mean(Qless_accuracies):.2f}')

