import numpy as np

def mse(self, label, output):
    return np.mean(np.power(label - output , 2))

def cross_entropy(self, label, output):
    return  np.sum(label * np.log2(output)) * (-1)
    
def mse_derivative(self, label, output):
    n = len(label)
    return (label - output)*(2/n)

def cross_entropy_derivative(self, label, output):
    return output - label

def L1(self, network_weights):
    penalty = 0
    for weights in network_weights:
        penalty += np.sum(np.where(weights, abs(weights), abs(weights)))
    return penalty

def L2(self, network_weights):
    penalty = 0
    for weights in network_weights:
        penalty += np.sum(np.where(weights, weights**2, weights**2))
    return penalty/2

def get_loss_function(name):
    if name == "mse":
        return mse
    elif name == "cross_entropy":
        return cross_entropy

def get_loss_derivative(name):
    if name == "mse":
        return mse_derivative
    elif name == "cross_entropy":
        return cross_entropy_derivative

def get_penalty_function(name):
    if name == "L1":
        return L1
    elif name == "L2":
        return L2
    elif name == "None":
        return None