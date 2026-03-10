import numpy as np


class Layer:
    """
    One layer of a neural network.

    Supports forward and backward passes to calculate weights.

    Attributes
    ----------
    weights : np.array
        The weights of the layer.
    bias : float
        The bias of the layer.
    """
    
    def __init__(self):
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass for a given input.

        Parameters
        ----------
        input : np.array
            The potentially multi-dimensional input to feed into the layer.

        Returns
        -------
        np.array
            The result of passing input through the layer.
        """
        pass

    def backward(self) -> None:
        pass

class Linear(Layer):
    def __init__(self, input_size: int, output_size :int):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.rand()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.activations = input @ self.weights + self.bias
        return self.activations
    
    def backward(self, loss: np.ndarray, input: np.ndarray) -> np.ndarray:
        pass 

class Conv(Layer):
    def __init__(self, num_filters:int, size:int, stride:int, padding:int):
        self.size = size
        self.stride = stride
        self.padding = padding

    def forward(self):
        pass

    def backward(self):
        pass

class MaxPool(Layer):
    def __init__(self, size:int):
        self.size = size

    def forward(self, x:np.ndarray):
        width = x.shape[0]
        height = x.shape[1]

        assert not (width % self.size or height % self.size)
        output = np.empty((width//self.size, height//self.size))
        
        for i in range(width//self.size):
            for j in range(height//self.size):
                output[i][j] = np.max(x[i*self.size : (i+1)*self.size, 
                                        j*self.size : (j+1)*self.size])
        return output

    def backward(self):
        pass

class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, x:np.ndarray):
        return np.clip(x, a_min=0, a_max=None)

    def backward(self):
        pass

class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass   

class SoftMax(Layer):
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

def CrossEntropyLoss():
    y = None # 2d array of 1 if correct prediction for each class, Shape (batch_size, num_classes)
    p = None # 2d array of probabilities for the correct prediction, Shape (batch_size, num_classes)
    
    p = np.clip(p, a_min=0.0000001, a_max=None)
    return np.sum(np.multiply(y, np.log(p))) / y.shape[0]

class Net():
    """
    Custom neural network architecture that supports convolutional layers
    """
    def __init__(self, layers:list[Layer]):
        self.layers = layers

    def forward_pass(self, x:np.ndarray):
        pass

    def backward_pass(self, x:np.ndarray):
        pass

    def train(self, x:np.ndarray):
        pass

if __name__ == '__main__':
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
        Linear(84, 26),
        SoftMax(),
    ]
    
    net = Net(layers)

    rng = np.random.default_rng() 
    test_arr = rng.random(size=(4, 4))
    test_arr = test_arr - 0.75
    print(test_arr)
    print('\n')

    pool = MaxPool(2)
    pooled = pool.forward(test_arr)

    print(pooled)
    print('\n')

    relu = ReLU()
    relued = relu.forward(pooled)

    print(relued)
    print('\n')


