import numpy as np

#make cnn

class Net():
    """
    Custom neural network architecture that supports convolutional layers
    """
    def __init__(self):
        pass

    def convolutional(self, x:np.ndarray, size:int, stride:int, padding:int):
        pass

    def fully_connected(self, x:np.ndarray, in_size:int, out_size:int):
        pass

    def relu(self, x:np.ndarray):
        pass

    def maxpool(self, x:np.ndarray):
        pass
        
    def forward_pass(self, x:np.ndarray):
        pass

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
    def __init__(self, layer_size: int):
        self.weights = np.random.randn(layer_size)
        self.bias = np.random.rand()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.activations = input @ self.weights + self.bias
        return self.activations
    
    def backward(self, loss: np.ndarray, input: np.ndarray) -> np.ndarray:
        pass 