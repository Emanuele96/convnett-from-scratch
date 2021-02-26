import numpy as np
import activations

class FC_layer():
    def __init__(self, input_size, output_size, weight_init_range, activation):
        self.type = "FC"
        self.activation_name = activation
        self.shape = (input_size, output_size)
        self.activation = activations.get_activation_function(activation)
        self.d_activation = activations.get_activation_derivative(activation)
        self.input = None
        self.output = None
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size=(input_size, output_size))
        #self.bias = np.zeros((1,output_size))
        self.bias = np.random.rand(1,output_size)
        self.weights_grads = np.zeros(self.weights.shape)
        self.bias_grads = np.zeros(self.bias.shape)
        

    def forward(self, input_activations):
        # Dot product of input with W plus bias. Cache, activate and return
        output = np.dot(input_activations, self.weights) + self.bias
        # Cache the weighted outputs and inputs
        self.output = output
        self.input = input_activations
        # Pass the output throug the activation function
        output = self.activation(self, output)
        return output
    
    def backward(self, jacobian_L_Z):
        # Get the jacobian linking the loss with respect of this layer output from the previous layer.
        # PURPOSE: Calculate the weights gradients, the bias gradient and the input_loss
        #           that will be passed to the previous activation layer and so on, up to layer previous input
        Y = self.input
        # Create the jacobian J_Z_sum with the layer cached outputs and the derivative of activation function
        jacobian_Z_sum = self.create_jacobian_Z_sum()

        # Find the Weights gradients jacobian_L_W
        # Compute the simple jacobian linking the outputs and the weights
        simp_jacobian_Z_W = np.outer(Y, jacobian_Z_sum.diagonal())
        # Then compute the jacobian linking the loss to the weights
        jacobian_L_W = jacobian_L_Z * simp_jacobian_Z_W

        # Calculate the input layer loss jacobian_L_Y
        # by doing dot product of output layer loss and the weigths matrix transposed (so to invert M N to N M, where M < N, we go the other way around)
        jacobian_Z_Y = np.dot(jacobian_Z_sum ,self.weights.T)
        jacobian_L_Y = np.dot( jacobian_L_Z, jacobian_Z_Y)
        

        # Bias loss is the as the output loss --> the bias influence on the loss == layer activation output influence on the loss
        jacobian_L_B = jacobian_L_Z

        # Now save the bias loss and weight loss (representing the calculated gradiants).
        # This will be updated at the end of the batch, or SGD
        self.weights_grads =self.weights_grads + jacobian_L_W
        self.bias_grads = self.bias_grads + jacobian_L_B
        
        #Finally return the calculated input loss --> this will be the output loss of the next layer
        return jacobian_L_Y

    def create_jacobian_Z_sum(self):
        return np.identity(self.output[0].size) * self.d_activation(self, self.output)

    def update_gradients(self, learning_rate, gradient_avg_factor = 1):
        #Update gradients, usefull when doing batch learning
        # Get the avg of the gradients (for SGD divide by 1, else divide by batchsize)
        ## UPDATE: removed the division by batchsize: Implemented this factor in the learning rate
        #self.weights_grads = self.weights_grads / gradient_avg_factor
        #self.bias_grads = self.bias_grads / gradient_avg_factor

        # Update weights and biases
        self.weights -= learning_rate * self.weights_grads
        self.bias -= learning_rate * self.bias_grads
        self.weights_grads = np.zeros(self.weights.shape)
        self.bias_grads = np.zeros(self.bias.shape)


    def __str__(self):
        return "FC Layer type size = " + str(self.weights.shape) + " with activation = " + self.activation_name

class conv2D():
    def __init__(self, input_size, n_kernels, kernel_shape,  strides, modes, weight_init_range, activation):
        self.type = "conv2D"
        self.activation_name = activation
        self.shape = (n_kernels, input_size, kernel_shape[0], kernel_shape[1])
        self.activation = activations.get_activation_function(activation)
        self.d_activation = activations.get_activation_derivative(activation)
        self.strides = strides
        self.modes = modes
        #self.input = None
        #self.output = None
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size= self.shape)
        #self.bias = np.zeros((1,output_size))
        self.weights_grads = np.zeros(self.weights.shape)

    def forward():
        return -1
    
    def backward():
        return -1
    
    def __str__(self):
        return "Conv 2D Layer type with "+  str(self.shape[0]) +" kernels of shape = " + str(self.shape[1:]) +" strides= " + str(self.strides) + " modes= " + str(self.modes) +" with activation = " + self.activation_name

class softmax():
    def __init__(self, size):
        self.size = size
        self.shape = (1, size)
        self.type = "softmax"
        self.activation_function = activations.softmax

    def forward(self, input_data):
        return  self.activation_function(self, input_data)

    def backward(self, jacobian_L_S, softmaxed_network_output):
        # Create jacobian of derivate of softmax
        jacobian_soft = self.compute_j_soft(softmaxed_network_output)    
        # Compute jacobian linking Loss to output 
        jacobian_L_Z = np.dot(jacobian_L_S, jacobian_soft)
        return jacobian_L_Z

    def compute_j_soft(self, S):
        S = np.squeeze(S)
        n = len(S)
        j_soft = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    j_soft[i][j] = S[i] - S[i]**2
                else:
                    j_soft[i][j] = -S[i]*S[j]
        return j_soft

    def __str__(self):
        return "Softmax Layer of size = " + str(self.size)

