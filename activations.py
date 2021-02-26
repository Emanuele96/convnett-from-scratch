import numpy as np

def linear(self, x):
    return x

def relu(self, x):
    return np.maximum(x, 0)

def tanh(self,x):
    return np.tanh(x)

def sigmoid(self, x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def linear_derivative(self, x):
    return 1

def relu_derivative(self, x):
    return np.where(x > 0, 1.0, 0.0)

def tanh_derivative(self, x):
    return 1 - np.tanh(x)**2

def sigmoid_derivative(self, x):
    return sigmoid(self, x)*(1-sigmoid(self, x))

def get_activation_function(name):
    if name == "linear":
        return linear
    elif name == "relu":
        return relu
    elif name == "sigmoid":
        return sigmoid
    elif name == "tanh":
        return tanh

def get_activation_derivative(name):
    if name == "linear":
        return linear_derivative
    elif name == "relu":
        return relu_derivative
    elif name == "sigmoid":
        return sigmoid_derivative
    elif name == "tanh":
        return tanh_derivative

def softmax(self, x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / np.sum(e_x)




