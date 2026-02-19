from emnist import extract_training_samples, extract_test_samples
import numpy as np

# If getting 'zipfile.BadZipFile: File is not a zip file'
# Download the emnist dataset from the website, rename gzip.zip to emnist.zip, and move it to ~/.cache/emnist

def train(X, y):
    pass

if __name__ == '__main__':
    train_images, train_labels = extract_training_samples('letters')
    test_images, test_labels = extract_test_samples('letters')

    print(f'Train images shape: {train_images.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Test images shape: {test_images.shape}')
    print(f'Test labels shape: {test_labels.shape}')

