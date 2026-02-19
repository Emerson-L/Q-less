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

class Net(torch.nn.Module):
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


if __name__ == '__main__':
    train_images, train_labels = extract_training_samples('letters')
    test_images, test_labels = extract_test_samples('letters')

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

    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    #dataiter = iter(testloader)
    #images, labels = next(dataiter)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

