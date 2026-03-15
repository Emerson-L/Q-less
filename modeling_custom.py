"""
modeling_custom.py
For training, testing, and saving a model using Indy and Emerson's custom CNN architecture (TM)

Example usage to train model:
python modeling_custom.py --train True --dataset combined --n_augments_rotation 3 --n_epochs 3 --model_path ./models/model.pkl

Example usage to test an existing model on Qless test set:
python modeling_custom.py --model_path model_combined_10_aug3r_noQ.pkl
"""

import numpy as np
import argparse
import pickle
from pathlib import Path

import wrangle
import visualize


class Layer:
    """
    One layer of a neural network.

    Supports forward and backward passes to make predictions and update weights.
    """
    
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass for a given input.

        Parameters
        ----------
        x : np.array
            The potentially multi-dimensional input to feed into the layer.

        Returns
        -------
        np.array
            The result of passing input through the layer.
        """
        pass

    def backward(self, grad: np.ndarray, lr:float) -> None:
        """
        Propogate the gradient backwards and update weights during training.

        Parameters
        ----------
        grad : np.ndarray
            The current gradient
        """
        pass 

class Linear(Layer):
    """
    Linear, fully connected layer

    Parameters
    ----------
    input_size: int
        The size of the input for the layer.
    output_size: int
        The number of neurons in the layer.

    Attributes
    ----------
    weights : np.array
        The weights of the layer.
    bias : float
        The bias of the layer.
    """
    def __init__(self, input_size: int, output_size :int):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)
        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return self.weights @ x + self.biases
    
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        self.weights -= grad.reshape(-1, 1) @ self.inputs.reshape(-1, 1).T * lr
        self.biases -= grad * lr
        return self.weights.T @ grad

class Conv(Layer):
    """
    Convolutional layer
    """
    def __init__(self, input_channels:int, num_filters:int, size:int, stride:int=1, padding:int=0):
        rng = np.random.default_rng(1) 
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.size = size
        self.stride = stride
        self.padding = padding

        init_st_dev = np.sqrt(2 / (input_channels * size * size)) # Kaiming He initialization: we initialize filters randomly with std root (2 / num_inputs)
        self.filters = [rng.normal(loc=0, scale=init_st_dev, size=(self.size, self.size)) for _ in range(num_filters)]

        self.input = None

    def forward(self, x:np.ndarray):
        assert x.shape[0] == self.input_channels
        self.input = x

        x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        example_conv = self.convolve(x[0], self.filters[0])
        output = np.zeros((self.num_filters, example_conv.shape[0], example_conv.shape[1]))
        for filter_idx, filter in enumerate(self.filters):
            for channel in range(self.input_channels):
                image = x[channel]
                output[filter_idx] += self.convolve(image, filter)

        return output
    
    def convolve(self, image:np.ndarray, filter:np.ndarray, full:bool=False):
        """
        Full convolution means with enough padding so that the edge of the filter goes to the very edge of the image
        """
        assert (image.shape[0] > filter.shape[0]) and (image.shape[1] > filter.shape[1])
        image = np.array(image)
        filter = np.array(filter)

        filter_width, filter_height = filter.shape[0], filter.shape[1]
        if full:
            image = np.pad(image, ((filter_width - 1, filter_height - 1), (filter_width - 1, filter_height - 1)))

        outdim_x = int(((image.shape[0] - filter_width) / self.stride) + 1)
        outdim_y = int(((image.shape[1] - filter_height) / self.stride) + 1)
    
        convolved = np.zeros((outdim_x, outdim_y))
        for i in range(outdim_x):
            for j in range(outdim_y):
                image_part = image[i*self.stride : (i*self.stride) + filter_width, 
                                   j*self.stride : (j*self.stride) + filter_height]
                #print(f'image_part shape: {image_part.shape}')
                #print(f'filter shape: {filter.shape}')
                convolved[i][j] = np.sum(image_part * filter)
        return convolved

    def backward(self, grad: np.ndarray, lr:float) -> np.ndarray:
        example_output_conv = self.convolve(grad[0], np.rot90(self.filters[0], k=2), full=True)
        output_grad = np.zeros((self.input_channels, example_output_conv.shape[0], example_output_conv.shape[1]))
        #print('output_grad shape')
        #print(output_grad.shape)
        for channel in range(self.input_channels):
            image_grad = grad[channel]
            #print('image_grad shape')
            #print(image_grad.shape)
            #print('1 channel self.input shape')
            #print(self.input[channel].shape)

            filter_grad = self.convolve(self.input[channel], image_grad) # get dl/df, the regular conv of in gradient and input
            #print('filter_grad shape')
            #print(filter_grad.shape)
            for filter_idx, filter in enumerate(self.filters):
                rotated_180_filter = np.rot90(filter, k=2)
                #print(filter.shape)
                #print(rotated_180_filter.shape)
                output_grad[channel] += self.convolve(image_grad, rotated_180_filter, full=True) # update grad here with full conv of grad and 180 rotated filter

                filter -= filter_grad * lr # update each filter with filter_grad

        #print('output grad shape')
        #print(output_grad.shape)
        return output_grad

class MaxPool(Layer):
    """
    Max pool layer
    """
    def __init__(self, size:int):
        self.size = size
        self.max_positions = None

    def forward(self, x:np.ndarray):
        if x.ndim == 2:
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        channels = x.shape[0]
        width = x.shape[1]
        height = x.shape[2]

        assert not (width % self.size or height % self.size)
        self.max_positions = np.empty((channels, width//self.size, height//self.size))
        output = np.empty((channels, width//self.size, height//self.size))
        
        for channel in range(channels):
            image = x[channel]
            for i in range(width//self.size):
                for j in range(height//self.size):
                    window = image[i*self.size : (i+1)*self.size, 
                                   j*self.size : (j+1)*self.size]
                    self.max_positions[channel, i, j] = np.argmax(window)
                    output[channel, i, j] = np.max(window)
        
        return output

    def backward(self, grad: np.ndarray, lr:float) -> np.ndarray:
        channels = grad.shape[0]
        width = grad.shape[1]
        height = grad.shape[2]
        
        output = np.empty((channels, width * self.size, height * self.size))
        for channel in range(channels):
            for i in range(width):
                for j in range(height):
                    pos_index = int(self.max_positions[channel, i // self.size, j // self.size])
                    pos = np.unravel_index(pos_index, (self.size, self.size))
                    output[channel, (i * self.size) + pos[0], (j * self.size) + pos[1]] = grad[channel, i, j]

        return output

class ReLU(Layer):
    """
    ReLU layer
    """
    def __init__(self):
        self.output = None

    def forward(self, x:np.ndarray):
        self.output = np.clip(x, a_min=0, a_max=None)
        return self.output

    def backward(self, grad: np.ndarray, lr:float) -> np.ndarray:
        return np.where(self.output, grad, 0)

class Flatten(Layer):
    """
    Flatten layer
    """
    def __init__(self):
        self.input_shape = None

    def forward(self, x:np.ndarray):
        self.input_shape = x.shape
        return x.reshape(-1)

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        return grad.reshape(self.input_shape)

class SoftMax(Layer):
    """
    Softmax layer
    """
    def __init__(self):
        pass

    def forward(self, x:np.ndarray):
        exp = np.exp(x)
        self.output = exp / np.sum(exp)
        return self.output

    def backward(self, grad: np.ndarray, lr:float) -> np.ndarray:
        return self.output - grad
        

def CrossEntropyLoss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the cross entropy loss for a batch of data.

    Parameters
    ----------
    probs : np.ndarray
        The predicted probabilities between 0 and 1, with length num_classes
    labels : np.ndarray
        The true labels for the data with values 0 or 1, with length num_classes

    Returns
    -------
    float
        The average loss across all data points in the batch.
    """
    p = np.clip(probs, a_min=1e-4, a_max=None)
    return -1 * np.sum(labels * np.log(p)) / labels.shape[0]

class Net():
    """
    Custom neural network architecture that supports convolutional layers
    """
    def __init__(self, layers:list[Layer]):
        self.layers = layers
        self.learning_rate = 0.0001

    def forward_pass(self, x:np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
            # print(f'After forward {str(type(layer)):<28}: {x.shape}')
            # print(x)
        return x

    def backward_pass(self, grad:np.ndarray):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, lr=self.learning_rate)
            # print(f'After backward {str(type(layer)):<28}: {grad.shape}')
            # print(grad)
        return grad

    def train(self, x_train:np.ndarray, y_train:np.ndarray, n_epochs:int, model_path:str):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        losses = []
        for epoch in range(n_epochs):
            running_loss = 0

            for i, (image, label) in enumerate(zip(x_train, y_train)):
                preds = self.forward_pass(image)

                correct = np.zeros((len(preds)))
                correct[label] = 1

                self.backward_pass(correct)

                running_loss += CrossEntropyLoss(preds, correct)

                if i % 200 == 199:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    losses.append(running_loss)
                    running_loss = 0
                    #break

        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print('Trained, saved model')
        return losses
    
def load_and_test(x_test:np.ndarray, y_test:np.ndarray, model_path:str, plot_wrong_predictions:bool=False):
    """
    Load a model and evaluate it on the given test set

    Parameters
    ----------
    x_test : np.ndarray
        array of test images
    y_test : np.ndarray
        array of test labels
    model_path : str
        path to model .pkl to load and test
    plot_wrong_predictions : bool
        whether to plot predictions that were incorrect 

    Returns
    -------
    accuracy : float
        model accuracy on test set
    """
    with open(model_path, 'rb') as f:
        net = pickle.load(f)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    correct = 0
    total = 0
    for image, label in zip(x_test, y_test):
        image = np.array(image)
        probs = net.forward_pass(image)
        pred = np.argmax(probs)
        if pred == label:
            correct += 1
        total += 1
        if plot_wrong_predictions:
            # may break because these are singular image rather than arrays of multiple
            visualize.plot_predictions_with_letter(image, pred, label, probs=probs)
    accuracy = 100 * correct / total
    return accuracy

def build_model():
    layers = [
        # Conv(1, 2, 5),
        # ReLU(),
        # MaxPool(2),
        # Conv(2, 16, 5),
        # ReLU(),
        MaxPool(2),
        Flatten(),
        Linear(196, 120), #was 256, 120
        ReLU(),
        Linear(120, 84),
        ReLU(),
        Linear(84, 25),
        SoftMax(),
    ]

    layers = [
        MaxPool(2),
        Flatten(),
        Linear(196, 120),
        ReLU(),
        Linear(120, 25),
        SoftMax(),
    ]

    return Net(layers)


def make_test_arr():
    rng = np.random.default_rng(111) 
    test_arr_size = 28
    test_arr = np.round(rng.random(size=(test_arr_size, test_arr_size)), decimals=2)
    test_arr = test_arr - 0.75
    return np.reshape(test_arr, (1, test_arr_size, test_arr_size))

def test():
    # Original test layers
    # test_layers = [
    #     Flatten((28, 28)),
    #     Linear(784, 120),
    #     ReLU(),
    #     Linear(120, 25),
    #     SoftMax(),
    # ]

    # With a MaxPool
    test_layers = [
        MaxPool(2),
        Flatten(),
        Linear(196, 120),
        ReLU(),
        Linear(120, 25),
        SoftMax(),
    ]

    test_net = Net(test_layers)

    test_arr = make_test_arr()

    #images, labels = wrangle.load_qless_test_data('./assets/letter_images_testset/IMG_3296')
    x_train, y_train, x_test, y_test = wrangle.load_data_splits_from_args('chars74k', 0)
    images = x_train
    labels = y_train

    # images = np.array(images[:1000])
    # labels = np.array(labels[:1000])

    print(images[0])
    print(labels[0])

    # Include for 2 channel input
    # test_arr = np.concatenate((test_arr, test_arr * 1.5), axis=0)
    import math
    losses = []
    running_losses = []
    for epoch in range(3):
        pred_letters = []
        for i, (image, label) in enumerate(zip(images, labels)):

            #image = np.array(image / 255)
            image = np.array(image)

            #print('Forwarding')
            preds = test_net.forward_pass(image)
            pred_letters.append(np.argmax(preds))

            correct = np.zeros((len(preds)))
            correct[label] = 1

            #print('Backwarding')
            test_net.backward_pass(correct)

            # break for testing one forward and back
            # break

            #print(preds)
            loss = CrossEntropyLoss(preds, correct)
            running_losses.append(loss)
            if i % 100 == 99:
                print(loss)
            #if math.isnan(loss):
            #    break

        avg_running_loss = np.mean(running_losses)
        print(avg_running_loss)
        losses.append(avg_running_loss)
        running_losses = []
        
    #print(list(labels))
    #print(pred_letters)

    visualize.plot_loss_curve(losses)


if __name__ == '__main__':
    #test()

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--n_augments_rotation', type=int)
    parser.add_argument('-e', '--n_epochs', type=int)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-t', '--train', type=bool)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    args = parser.parse_args()

    if not Path(args.model_path).suffix == '.pkl':
        args.model_path = Path(args.model_path).with_name(Path(args.model_path).stem + '.pkl')
    if not Path(args.model_path).exists() and not args.train:
        raise ValueError(f'Model path {args.model_path} does not exist. Call with --train flag to train')

    if args.train:
        x_train, y_train, x_test, y_test = wrangle.load_data_splits_from_args(args.dataset, args.n_augments_rotation)
        
        net = build_model()
        losses = net.train(x_train, y_train, args.n_epochs, args.model_path)
        visualize.plot_loss_curve(losses)

        accuracy = load_and_test(x_test, y_test, args.model_path)
        print(f'Accuracy on test set: {accuracy:.2f}%')

    #eval on qless test set
    



