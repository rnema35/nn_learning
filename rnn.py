import numpy as np
from functions import *

class RNN_Layer: 
    def __init__(self, input_dim, hidden_dim, output_dim, input_activation, output_activation):
        #let input = n, hidden = k, output = m
        self.W_ax = np.random.rand(hidden_dim, input_dim) * 0.1 # k x n matrix
        self.W_aa = np.random.rand(hidden_dim, hidden_dim) * 0.1 # k x k matrix
        self.W_ay = np.random.rand(output_dim, hidden_dim) * 0.1 # m x k matrix
        self.b_a = np.zeros((hidden_dim, 1)) # k x 1 vector
        self.b_y = np.zeros((output_dim, 1)) # m x 1 vector
        self.input_activation = input_activation
        self.output_activation = output_activation

    def get_activation(self, z, activation_function):
        if activation_function == 'hidden':
            activation = self.input_activation
        elif activation_function == 'output':
            activation = self.output_activation

        if activation == 'sigmoid':
            return sigmoid(z)
        elif activation == 'relu':
            return relu(z)
        elif activation == 'linear':
            return linear_activation(z)
        else:
            print('we should not be here!')
            return 1

class RNN: 
    def __init__(self, layer, passes):
        self.layer = layer
        self.passes = passes # is this needed? won't this be the amount of times we feed it data
        self.cache = {}

    #not sure if this correct --> how do you even mini batch train a recurrent neural net??
    #you can't shuffle it, you have to do it by splitting it in different windows of time
    #or you make a lot of data to train it on
    def mini_batch_train(self, training_data, test_data, epochs, learning_rate, batch_size = 32):
        n = len(training_data)
        print("<------------------> Training Mini Batch <------------------>")

        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]    

            for mini_batch in mini_batches:
                self.update_mini_batch_params(mini_batch, learning_rate)

            loss = self.quad_cost_loss(training_data)
            accuracy = self.compute_accuracy(training_data)
            print(f"Epoch {i + 1}/{epochs}, training loss: {loss: .4f}, accuracy: {accuracy: .4f}")

        print("<------------------> Test <------------------>")
        test_loss = self.quad_cost_loss(test_data)
        test_acc = self.compute_accuracy(test_data)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    
    def feedforward(self, x):
        
        layer = self.layer
        a = np.zeros_like(layer.b_a) # t_0 activation is of 0s

        for l in range(self.passes): 
            a_prev = a
            
            #fix the way we get x
            z = np.dot(layer.W_ax, x) + np.dot(layer.W_aa, a_prev) + layer.b_a
            a = layer.get_activation(z, 'hidden')
            
            y_unactivated = np.dot(layer.W_ay, a) + layer.b_y
            y = layer.get_activation(y_unactivated, 'output')

        return y
    
    