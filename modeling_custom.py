import numpy as np
import wrangle


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

    def backward(self, grad: np.ndarray) -> None:
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
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = x @ self.weights + self.biases
        return self.activations
    
    def backward(self, grad: np.ndarray, input: np.ndarray, lr: float) -> np.ndarray:
        self.weights -= grad @ input.T * lr
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

        self.outdim_x = None
        self.outdim_y = None

    def forward(self, x:np.ndarray):
        assert x.shape[0] == self.input_channels

        x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.outdim_x = int(((x[0].shape[0] - self.size) / self.stride) + 1)
        self.outdim_y = int(((x[0].shape[1] - self.size) / self.stride) + 1)

        output = np.zeros((self.num_filters, self.outdim_x, self.outdim_y))
        for filter_idx, filter in enumerate(self.filters):
            for channel in range(self.input_channels):
                image = x[channel]
                output[filter_idx] += self.convolve(image, filter)

        return output
    
    def convolve(self, image:np.ndarray, filter:np.ndarray):
        convolved = np.zeros((self.outdim_x, self.outdim_y))
        for i in range(self.outdim_x):
            for j in range(self.outdim_y):
                image_part = image[i*self.stride : (i*self.stride)+self.size, 
                                   j*self.stride : (j*self.stride)+self.size]
                convolved[i][j] = np.sum(image_part * filter)
        return convolved

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass 
        # Update filter weights

class MaxPool(Layer):
    """
    Max pool layer
    """
    def __init__(self, size:int):
        self.size = size

    def forward(self, x:np.ndarray):
        channels = x.shape[0]
        width = x.shape[1]
        height = x.shape[2]

        assert not (width % self.size or height % self.size)
        output = np.empty((channels, width//self.size, height//self.size))
        
        for channel in range(channels):
            image = x[channel]
            for i in range(width//self.size):
                for j in range(height//self.size):
                    output[channel, i, j] = np.max(image[i*self.size : (i+1)*self.size, 
                                                         j*self.size : (j+1)*self.size])
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass 

class ReLU(Layer):
    """
    ReLU layer
    """
    def __init__(self):
        self.output = None

    def forward(self, x:np.ndarray):
        self.output = np.clip(x, a_min=0, a_max=None)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(self.output, grad, 0)

class Flatten(Layer):
    """
    Flatten layer
    """
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def forward(self, x:np.ndarray):
        return x.reshape(-1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
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

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.output - grad
        

def CrossEntropyLoss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the cross entropy loss for a batch of data.

    Parameters
    ----------
    probs : np.ndarray
        The predicted probabilities between 0 and 1, with length num_classes
    labels : np.ndarray
        The true labels for the data with valyes 0 or 1, with length num_classes

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

    def forward_pass(self, x:np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
            print(layer)
            # print(x)
            print(x.shape)
            #print('\n')
        return x

    def backward_pass(self, grad:np.ndarray):
        for layer in self.layers:
            grad = layer.backward(grad)
            print(layer)
            # print(grad)
            print(grad.shape)
            #print('\n')

    def train(self, x:np.ndarray, y:np.ndarray):
        pass

def build_model():
    layers = [
        Conv(1, 2, 5),
        ReLU(),
        MaxPool(2),
        Conv(2, 16, 5),
        ReLU(),
        MaxPool(2),
        Flatten(),
        Linear(256, 120),
        ReLU(),
        Linear(120, 84),
        ReLU(),
        Linear(84, 25),
        SoftMax(),
    ]

    return Net(layers)


def make_test_arr():
    rng = np.random.default_rng(111) 
    test_arr_size = 28
    test_arr = np.round(rng.random(size=(test_arr_size, test_arr_size)), decimals=2)
    test_arr = test_arr - 0.75
    return np.reshape(test_arr, (1, test_arr_size, test_arr_size))

if __name__ == '__main__':
    net = build_model()
    
    test_layers = [
        Flatten(),
        Linear(784, 120),
        ReLU(),
        Linear(120, 25),
        SoftMax(),
    ]
    test_net = Net(test_layers)

    test_arr = make_test_arr()

    images, labels = wrangle.load_qless_test_data('./assets/letter_images/IMG_3296')

    # Include for 2 channel input
    # test_arr = np.concatenate((test_arr, test_arr * 1.5), axis=0)
    
    # print('Input array')
    # print(test_arr)
    # print('\n')

    # print('Testing maxpool and relu')
    # pool = MaxPool(2)
    # pooled = pool.forward(test_arr)
    # print(pooled)
    # print('\n')

    # relu = ReLU()
    # relued = relu.forward(pooled)
    # print(relued)
    # print('\n')

    # print('Testing convolution')
    # conv = Conv(input_channels=2, num_filters=2, size=3, stride=1, padding=0)
    # conv_arr = conv.forward(test_arr)
    # print('Convolved array')
    # print(conv_arr)
    # print(conv_arr.shape)
    # print('\n')

    # print('Testing softmax')
    # softmaxed = SoftMax().forward(np.array([1, 2, 3, 4]))
    # print(softmaxed)
    # print(np.sum(softmaxed))

    # print('Testing flatten')
    # flattened = Flatten().forward(test_arr)
    # print(flattened)

    print('Testing full forwarding')
    for image in images:
        test_net.forward_pass(image)
        #test_net.backward_pass()



