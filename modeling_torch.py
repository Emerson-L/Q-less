from emnist import extract_training_samples, extract_test_samples
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import cv2 as cv

import visualize
import utils

# If getting 'zipfile.BadZipFile: File is not a zip file'
# Download the emnist dataset from the website, rename gzip.zip to emnist.zip, and move it to ~/.cache/emnist

NUM_DICE = 12 # Should centralize config stuff like this at some point
BATCH_SIZE = 32
MODEL_PATH = './letters_model.pth'

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
    
def train(trainloader:DataLoader, model_path:str) -> None:
    """
    Trains and saves model. Derived from pytorch CNN tutorial

    Parameters
    ----------
    trainloader : torch DataLoader
        DataLoader with training dataset
    model_path : str
        path to save trained model to
    """
    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    torch.save(net.state_dict(), model_path)

    print('Finished training, saved model')


def load_and_test(testloader:DataLoader, model_path:str, plot_predictions:bool=False) -> float:
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

    # for i in range(BATCH_SIZE):
    #     visualize.plot_image(images[i][0])

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
            if plot_predictions:
                for i, (image, pred_idx) in enumerate(zip(images, predicted)):
                    letter = utils.numbers_to_letters([pred_idx])[0]
                    visualize.plot_image(image[0], letter=letter)
                    visualize.plot_probs(utils.numbers_to_letters(range(26)), softmaxed_outputs[i])

    accuracy = 100 * correct / total
    return accuracy


def load_emnist_data() -> tuple[DataLoader, DataLoader]:
    """
    Loads emnist letters dataset into train and test DataLoaders

    Returns
    -------
    train_loader : torch DataLoader
        DataLoader with train set
    test_loader : torch DataLoader
        DataLoader with test set
    """
    train_images, train_labels = extract_training_samples('letters')
    test_images, test_labels = extract_test_samples('letters')

    train_labels = train_labels - 1
    test_labels = test_labels - 1

    print(f'Train images shape: {train_images.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Test images shape: {test_images.shape}')
    print(f'Test labels shape: {test_labels.shape}')

    x_train = torch.tensor(train_images).unsqueeze(1).float() 
    y_train = torch.tensor(train_labels).long()

    x_test = torch.tensor(test_images).unsqueeze(1).float()
    y_test = torch.tensor(test_labels).long()

    x_train /= 255.0
    x_test /= 255.0

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    images = np.zeros((NUM_DICE, 28, 28), dtype=np.uint8)
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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return loader

if __name__ == '__main__':
    # trainloader, testloader = load_emnist_data()
    # train(trainloader, MODEL_PATH)

    #accuracy = load_and_test(testloader, MODEL_PATH)
    #print(f'Accuracy on EMNIST test set: {accuracy:.2f}%')

    # Test out our model on real Qless data
    test_images_dirs = ['./assets/letter_images/IMG_3296/',
                         './assets/letter_images/IMG_3297/',
                         './assets/letter_images/IMG_3298/',
                         './assets/letter_images/IMG_3299/',]
    for dir in test_images_dirs:
        Qless_testloader = load_qless_test_data(dir)
        Qless_accuracy = load_and_test(Qless_testloader, MODEL_PATH, plot_predictions=True)
        print(f'Accuracy on {dir}: {Qless_accuracy:.2f}%')


