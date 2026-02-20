from emnist import extract_training_samples, extract_test_samples
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import visualize

# If getting 'zipfile.BadZipFile: File is not a zip file'
# Download the emnist dataset from the website, rename gzip.zip to emnist.zip, and move it to ~/.cache/emnist

BATCH_SIZE = 32

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


def load_emnist_data():
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

    #visualize.plot_image(train_images[0])

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

def load_and_test(testloader:DataLoader, model_path:str) -> float:
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
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    MODEL_PATH = './letters_model.pth'

    trainloader, testloader = load_emnist_data()
    print(type(testloader))

    #train(trainloader, MODEL_PATH)

    accuracy = load_and_test(testloader, MODEL_PATH)
    print(f'Accuracy of the network on test images: {accuracy:.2f}%')


    # Try predicting on our images next

