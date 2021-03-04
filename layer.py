import numpy as np
import math
import activations

class FC_layer():
    def __init__(self, input_size, output_size, weight_init_range, activation, debug):
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
        self.debug = debug

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
    def __init__(self, input_shape, n_kernels, kernel_shape,  strides, modes, weight_init_range, activation, debug):
        self.type = "conv2D"
        self.input_shape = input_shape
        self.activation_name = activation
        #Kernel stack shape for the layer (N, I, K_x, K_y)
        self.kernel_shape = (n_kernels, input_shape[0], kernel_shape[0], kernel_shape[1])
        self.activation = activations.get_activation_function(activation)
        self.d_activation = activations.get_activation_derivative(activation)
        self.strides = strides
        self.modes = modes
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size= self.kernel_shape)
        self.weights_grads = np.zeros(self.weights.shape)
        self.p_x_start, self.p_x_stop, self.p_y_start, self.p_y_stop = self.calculate_padding()
        self.output_shape = self.calculate_output_shape()
        self.cached_calculation = {}
        self.cache_weights_input_output_triplet_locations()
        self.cached_output = None
        self.debug = debug

        
        
        '''print("###########################")
        a = np.random.randint(1,4,(6,6))
        print(a)
        padded_a = self.apply_zero_padding(a)
        print(padded_a)
        print("kernel shape", (self.kernel_shape[2], self.kernel_shape[3]))
        print("input shape", a.shape)
        print("padded shape", padded_a.shape)
        print("###########################")'''

    def cache_weights_input_output_triplet_locations(self):
        placeholder_input = np.zeros(self.input_shape)
        array = placeholder_input[0]
        kernel = self.weights[0][0]
        stride_x_pointer = 0
        while(stride_x_pointer + kernel.shape[0] - 1 <= array.shape[0] - 1):
            stride_y_pointer = 0
            #while the kernel does not go over the x-akse of the array
            while(stride_y_pointer + kernel.shape[1] -1 <= array.shape[1] - 1):
                #while the kernel does not go over the x-akse of the array
                #cache all  touched weights and input for each kernel (output or Coordinates??)
                for row in range(kernel.shape[0]):
                    for column in range(kernel.shape[1]):
                        # Cache coordinate only: (weight, input) --> output
                        #format: key ((weight_x_pos, weight_y_pos), (input_x_pos, input_y_pos)) ---> (output_x_pos, output_y_pos)
                        conv_output_coordinate = (stride_x_pointer // self.strides[0], stride_y_pointer // self.strides[1])
                        self.cached_calculation[((row, column), (row + stride_x_pointer , column + stride_y_pointer))] = conv_output_coordinate
                        #Cache weight coordinate and input/output values
                # Update the stride long the y-axis
                stride_y_pointer += self.strides[1]
            #update the stride long the x-axis
            stride_x_pointer += self.strides[0]
        #End of convolution
            

    def forward(self, input_feature_maps):
        #reset the cached calculations from the previous forward pass
        self.cached_calculation = {}
        output = np.zeros(self.output_shape)
        #Apply padding
        input_feature_maps = self.apply_zero_padding(input_feature_maps)
        for i in range(0, self.kernel_shape[0]):
            #for each kernel stack
            kernel_stack = self.weights[i]
            for j in range(0, self.kernel_shape[1]):
                #for each kernel in the kernel stack (or input channel)
                kernel = kernel_stack[j]
                array = input_feature_maps[j]
                stride_x_pointer = 0
                conv_counter = 1
                if self.debug:
                    print("**** NEW CONVOLUTION ****")
                while(stride_x_pointer + kernel.shape[0] - 1 <= array.shape[0] - 1):
                    stride_y_pointer = 0
                    #while the kernel does not go over the x-akse of the array
                    while(stride_y_pointer + kernel.shape[1] -1 <= array.shape[1] - 1):
                        #while the kernel does not go over the x-akse of the array
                        #Get the snip of the array to apply convolution on
                        array_snip = array[stride_x_pointer: stride_x_pointer + kernel.shape[0], stride_y_pointer: stride_y_pointer + kernel.shape[1]]
                        #apply convolution and get the result 
                        result = np.sum(np.multiply(array_snip, kernel))                            
                        #update the output tensor
                        conv_output_coordinate = (i, stride_x_pointer // self.strides[0], stride_y_pointer // self.strides[1])
                        output[conv_output_coordinate] += result
                        '''#cache all the results, touched weights and input for each kernel (output or Coordinates??)
                        for row in range(kernel.shape[0]):
                            for column in range(kernel.shape[1]):
                                # Cache coordinate only: (weight, input) --> output
                                #format: key ((kernel_stack_number, 2D_kernel_number, weight_x_pos, weight_y_pos), (input_channel, input_x_pos, input_y_pos)) ---> (feature_map_number, output_x_pos, output_y_pos)
                                self.cached_calculation[((i, j, row, column), (j, row + stride_x_pointer , column + stride_y_pointer))] = conv_output_coordinate
                                #Cache weight coordinate and input/output values
                                #ALTERNATIVE
                                # format: key ((kernel_stack_number, 2D_kernel_number, weight_x_pos, weight_y_pos), input_val) ---> output_val
                                #self.cached_calculation[((i, j, row, column), array_snip[row, column])] = result'''
                        if self.debug:
                            print("convolution nr ", conv_counter )
                            print("\narray_snip: \n", array_snip)
                            print("\nkernel: \n", kernel)
                            print("\nelementwise multiplication: \n", np.multiply(array_snip, kernel))
                            print("\nresult: ", result)
                        # Update the stride long the y-axis
                        stride_y_pointer += self.strides[1]
                        conv_counter+=1
                    #update the stride long the x-axis
                    stride_x_pointer += self.strides[0]
                #End of convolution
                if self.debug:
                    print("\n----REVIEW----\n")
                    print("Total convolutions: ", conv_counter)
                    print("\ninput_feature_map:\n ", array)
                    print("\napplied kernel:\n ", kernel)
                    print("\nconvolution result:\n ", output[i])
                    print("***********************************")
        #Cache input and output
        self.cached_output = output
        self.cached_input = input_feature_maps
        #Apply activation
        output = self.activation(self, output)
        return output
                
    
    def backward(self, jacobian_L_Z):
        #Reshape J_LZ from FC to Conv2D and pass through activation layer
        jacobian_L_Z = jacobian_L_Z.reshape(self.output_shape)
        jacobian_L_Z = self.d_activation(self, jacobian_L_Z)

        #Calculate J_LW
        jacobian_L_W = self.compute_gradients(jacobian_L_Z)
        self.weights_grads += jacobian_L_W

        #Calculate J_LX
        jacobian_L_Y = self.compute_J_LY(jacobian_L_Z)

        #Pass Jacobian L Y upstream
        return jacobian_L_Y
    
    def update_gradients(self, learning_rate):
        self.weights -= learning_rate * self.weights_grads
        self.weights_grads = np.zeros(self.weights.shape)

    def compute_gradients(self, jacobian_L_Z):
        grads = np.zeros(self.weights.shape)
        #Iterate through all the weights (4 dimension)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                for k in range(self.weights.shape[2]):
                    for l in range(self.weights.shape[3]):
                        #cached_data = {k: v for k,v in self.cached_calculation.items() if k[0] == (i,j,k,l)}
                        for key in self.cached_calculation.keys():
                            if key[0] == (k,l):
                                grads[(i,j,k,l)] += self.cached_input[j][key[1]] * jacobian_L_Z[i][self.cached_calculation[key]]
        return grads

    def compute_J_LY(self, jacobian_L_Z):
        jacobian_L_Y = np.zeros(self.input_shape)
        #Iterate through all the inputs (3 dimension)
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[2]):
                        #cached_data = {k: v for k,v in self.cached_calculation.items() if k[0] == (i,j,k,l)}
                        for key in self.cached_calculation.keys():
                            if key[1] == (i,j,k):
                                jacobian_L_Y[(i,j,k)] += self.weights[key[0]] * jacobian_L_Z[self.cached_calculation[key]]
        return jacobian_L_Y
    
    def calculate_output_shape(self):
        width = math.floor((self.input_shape[1] - self.kernel_shape[2] + self.p_x_start + self.p_x_stop)/self.strides[0] + 1)
        height = math.floor((self.input_shape[2] - self.kernel_shape[3] + self.p_y_start + self.p_y_stop)/self.strides[1] + 1 )
        print(width, height)
        return (self.kernel_shape[0], width, height)

    def calculate_padding(self):
        #Calculate padding long the x axis
        s = self.strides[0]
        f = self.kernel_shape[2]
        i = self.kernel_shape[2]
        if self.modes[0] == "full":
        #Every pixel must experience every weight of the kernel
            p_x_start = f - 1
            p_x_stop = f - 1
        elif self.modes[0] == "same":
        #Every pixel must experience the middle weight of the kernel
            p_x_start = math.floor((s*math.ceil(i/s)-i+f-s)/2)
            p_x_stop = math.ceil((s*math.ceil(i/s)-i+f-s)/2)
        else:
            p_x_start = 0
            p_x_stop = 0


        #Calculate padding long y axis
        s = self.strides[1]
        f = self.kernel_shape[3]
        i = self.kernel_shape[3]
        if self.modes[1] == "full":
        #Every pixel must experience every weight of the kernel
            p_y_start = f - 1
            p_y_stop = f - 1
        elif self.modes[1] == "same":
        #Every pixel must experience the middle weight of the kernel
            p_y_start = math.floor((s*math.ceil(i/s)-i+f-s)/2)
            p_y_stop = math.ceil((s*math.ceil(i/s)-i+f-s)/2)
        else:
            p_y_start = 0
            p_y_stop = 0


        return p_x_start, p_x_stop, p_y_start, p_y_stop
    
    def apply_zero_padding(self, input_feature_maps):
    # Apply zero padding to the input feature maps according to the modes, strides and kernel size
        padded_input_feature_maps = np.zeros((input_feature_maps.shape[0], input_feature_maps.shape[1] + self.p_x_start + self.p_x_stop, input_feature_maps.shape[2] + self.p_y_start + self.p_y_stop ))
        for channel in range(input_feature_maps.shape[0]):
            array = input_feature_maps[channel]
            #Create the background zero array
            padded_array = np.zeros((array.shape[0] + self.p_x_start + self.p_x_stop, array.shape[1] + self.p_y_start + self.p_y_stop))
            #Copy the array in the middle of the zero background
            padded_array[self.p_x_start:array.shape[0]+ self.p_x_start, self.p_y_start:array.shape[1]+ self.p_y_start] = array 
            #Save the array
            padded_input_feature_maps[channel] = padded_array
        return padded_input_feature_maps

    def __str__(self):
        return "Conv 2D Layer type with "+  str(self.kernel_shape[0]) +" kernels of shape = " + str(self.kernel_shape[1:]) +"input/output of shape" + str(self.input_shape)+"/" + str(self.output_shape) + "  strides= s" + str(self.strides) + " modes= " + str(self.modes) +" with activation = " + self.activation_name

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

