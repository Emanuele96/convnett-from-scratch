import numpy as np
import layer as l
import loss
import activations
import math
#from progress.bar import IncrementalBar
from tqdm import tqdm
import matplotlib.pyplot as plt


class Model():
    
    def __init__(self, cfg, input_shape):
        self.layers = list(())
        self.learning_rate = cfg["lr"]
        self.loss_function_name = cfg["loss_fun"]
        self.loss_fun = loss.get_loss_function(cfg["loss_fun"])
        self.loss_derivative = loss.get_loss_derivative(cfg["loss_fun"])
        self.input_shape = input_shape
        
    def add_layer(self, layer, input_shape, input_nodes, debug):

        if layer["type"] == "FC":
            self.layers.append(l.FC_layer(input_nodes, layer["size"], layer["weights_start"], layer["activation"], debug))
            output_nodes = layer["size"]
            output_shape = None
        elif  layer["type"] == "conv2D":
            if len(input_shape) == 2:
                tmp = np.zeros(input_shape)
                print("tmp", tmp.shape)
                tmp = np.reshape(tmp, (input_shape[0], math.ceil(math.sqrt(input_shape[1])),-1))
                input_shape = tmp.shape
            self.layers.append(l.conv2D(input_shape,  layer["number_kernels"], layer["kernel_shape"], layer["strides"], layer["modes"], layer["weights_start"], layer["activation"], debug))
            output_nodes = self.layers[-1].output_shape[0] * self.layers[-1].output_shape[1] * self.layers[-1].output_shape[2]
            output_shape = self.layers[-1].output_shape
        elif  layer["type"] == "conv1D":
            if len(input_shape) == 3:
                tmp = np.zeros(input_shape)
                tmp = tmp.reshape(input_shape[0], -1)
                input_shape = tmp.shape
            self.layers.append(l.conv1D(input_shape,  layer["number_kernels"], layer["kernel_shape"], layer["strides"], layer["modes"], layer["weights_start"], layer["activation"], debug))
            output_nodes = self.layers[-1].output_shape[0] * self.layers[-1].output_shape[1]
            output_shape = self.layers[-1].output_shape
        return output_shape, output_nodes

    def add_softmax(self):
        self.layers.append(l.softmax(self.layers[-1].shape[1]))

    def train(self, x_train, y_train, x_validate, y_validate, epochs, batch_size):
        #bar = IncrementalBar('Training epoch', max=epochs)
        samples = len(x_train)
        batches = math.ceil(samples/batch_size)
        losses = list(())
        validation_errors = list(())
        print("Training:")
        for i in tqdm(range(epochs)):
            # For each epoch --> go through the whole train set, one batch on the time
            for j in range(batches):
                # For each batch, go through "batch_size" samples 
                batch_loss = 0
                batch_samples = 0
                for k in range(batch_size):
                    # For each sample, propagate to the network.
                    # Then backpropagate network output and calculate gradients.
                    sample_nr = k + j * batch_size
                    omega = 0
                    if sample_nr == samples:
                        break
                    batch_samples += 1
                    # FORWARD PASS : Fetch the input data and propagate through the network
                    # This will be represent the input layer
                    network_output = x_train[sample_nr]
                    
                    for layer in self.layers:
                        if layer.type == "FC":
                            network_output = network_output.reshape(1,-1)
                        elif layer.type == "conv1D" and len(network_output.shape) == 3:
                            network_output = network_output.reshape(network_output.shape[0], -1)
                        elif layer.type == "conv2D" and len(network_output.shape) == 2:
                            network_output = network_output.reshape(network_output.shape[0], math.ceil(math.sqrt(network_output.shape[1])),-1)
                        network_output = layer.forward(network_output)
                    # Calculate the loss
                    loss = self.loss_fun(self,y_train[sample_nr], network_output)
                    batch_loss = batch_loss + loss
                    # BACKWARD PASS : Get the loss and backpropagate throught the network
                    jacobian_L_Z = self.loss_derivative(self,y_train[sample_nr],  network_output)

                    # backpropagate
                    for layer in reversed(self.layers):
                        if layer.type == "softmax":
                            jacobian_L_Z = layer.backward(jacobian_L_Z, network_output)
                        else:
                            jacobian_L_Z = layer.backward(jacobian_L_Z)

                # At the end of each batch, update gradients
                for layer in self.layers:
                    if layer.type == "FC" or layer.type == "conv2D":
                        layer.update_gradients(self.learning_rate)
                # Append the loss of the batch and validate the batch
                losses.append(batch_loss/batch_samples) 
            # end of an epoch
            validation_errors.append(self.validate(x_validate, y_validate))
            #bar.next()
        #bar.finish()
        return losses, validation_errors

    def predict(self, x):
        prediction = x
        for layer in self.layers:
            if layer.type == "FC":
                prediction = prediction.reshape(1,-1)
            elif layer.type == "conv1D" and len(prediction.shape) == 3:
                prediction = prediction.reshape(prediction.shape[0], -1)
            elif layer.type == "conv2D" and len(prediction.shape) == 2:
                prediction = np.reshape(prediction, (prediction.shape[0], math.ceil(math.sqrt(prediction.shape[1])),-1))
            prediction = layer.forward(prediction)
        return prediction
    
    def calculate_loss(self, label, prediction):
        return self.loss_fun(self, label, prediction)

    def validate(self, x_validate, y_validate):
        error = 0 
        for i in range(len(x_validate)):
            prediction = self.predict(x_validate[i])
            error += self.loss_fun(self, y_validate[i], prediction)
        return error/len(x_validate)

    def test(self,x_test, y_test):
        samples = len(x_test)
        loss = 0
        for i in range(samples): 
            loss += self.calculate_loss(y_test[i], self.predict(x_test[i]))
        return loss/samples

    def visualize_kernels(self):
        for layer in self.layers:
            if layer.type != "conv2D":
                continue
            print("#### Visualizing kernels Conv2D layer #####")
            for i in range(layer.weights.shape[0]):
                print("### Kernel stack nr. ", i, " ###")
                for j in range(layer.weights.shape[1]):
                    print("## Kernel nr. ", j, " ##")
                    print(layer.weights[i][j])
                    self.hinton(layer.weights[i][j])
                    plt.show()
            print("######################")

    
    def hinton(self, matrix, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        "From https://matplotlib.org/3.1.1/gallery/specialty_plots/hinton_demo.html"
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                    facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()
        

    def __str__(self):
        s = "\n***  Model Architecture *** \nInput Layer of size = " + str(self.input_shape)
        for layer in self.layers:
            s = s + "\n" + str(layer)
        s = s + "\nLearning rate : " + str(self.learning_rate)
        s = s + "\nLoss function : " + str(self.loss_function_name)
        s = s + "\n" + "**************************"
        return s
