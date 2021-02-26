import numpy as np
import layer
import loss
import activations
import math
from progress.bar import IncrementalBar


class Model():
    
    def __init__(self, cfg):
        self.layers = list(())
        self.learning_rate = cfg["lr"]
        self.loss_function_name = cfg["loss_fun"]
        self.loss_fun = loss.get_loss_function(cfg["loss_fun"])
        self.loss_derivative = loss.get_loss_derivative(cfg["loss_fun"])
        self.penalty_factor = cfg["penalty_factor"]
        self.penalty_function = loss.get_penalty_function(cfg["penalization"])
        self.penalty_function_name = cfg["penalization"]

    def add_layer(self,  input_size, output_size, weight_init_range, activation_function):
        self.layers.append(layer.FC_layer(input_size, output_size, weight_init_range, activation_function))
        return output_size

    def add_softmax(self):
        self.layers.append(layer.softmax(self.layers[-1].shape[1]))

    def train(self, x_train, y_train, x_validate, y_validate, epochs, batch_size):
        bar = IncrementalBar('Training epoch', max=epochs)
        samples = len(x_train)
        batches = math.ceil(samples/batch_size)
        losses = list(())
        validation_errors = list(())
        for i in range(epochs):
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
                        network_output = layer.forward(network_output)
                    
                    # Calculate the loss
                    loss = self.loss_fun(self,y_train[sample_nr], network_output)
                    batch_loss = batch_loss + loss
                    # BACKWARD PASS : Get the loss and backpropagate throught the network
                    jacobian_L_Z = self.loss_derivative(self,y_train[sample_nr],  network_output)

                    #Get the normalization penalty
                    if self.penalty_function_name != "None":
                        # Get all the weights of the network
                        network_weights = list(())
                        for layer in self.layers:
                            if layer.type == "FC":
                                network_weights.append(layer.weights)
                        omega = self.penalty_function(self, network_weights)
                        # Add the penalty to the loss function
                        jacobian_L_Z = jacobian_L_Z + self.penalty_factor*omega
                        
                    # backpropagate
                    for layer in reversed(self.layers):
                        if layer.type == "softmax":
                            jacobian_L_Z = layer.backward(jacobian_L_Z, network_output)
                        else:
                            jacobian_L_Z = layer.backward(jacobian_L_Z)

                # At the end of each batch, update gradients
                for layer in self.layers:
                    if layer.type == "FC":
                        layer.update_gradients(self.learning_rate)
                # Append the loss of the batch and validate the batch
                losses.append(batch_loss/batch_samples) 
            # end of an epoch
            validation_errors.append(self.validate(x_validate, y_validate))
            bar.next()
        bar.finish()
        return losses, validation_errors

    def predict(self, x):
        prediction = x
        for layer in self.layers:
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


    def __str__(self):
        s = "\n***  Model Architecture *** \nInput Layer of size = " + str(self.layers[0].shape[0])
        for layer in self.layers:
            s = s + "\n" + str(layer)
        s = s + "\nLearning rate : " + str(self.learning_rate)
        s = s + "\nLoss function : " + str(self.loss_function_name)
        s = s + "\nWeight regularization : " + str(self.penalty_function_name)
        if self.penalty_function_name != "None":
            s = s + "\nWeight regularization factor: " + str(self.penalty_factor)
        s = s + "\n" + "**************************"
        return s
