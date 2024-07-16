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

    def get_activation(self, z, activation_for):
        if activation_for == 'hidden':
            activation = self.input_activation
        elif activation_for == 'output':
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
    def __init__(self, layer):
        self.layer = layer
        #self.passes = passes # is this needed? won't this be the amount of times we feed it data
        self.cache = {}

    #will pass in an 2d array (3 if each time point has more than 1 feature, sticking with only 1 atm)
    #1000 sequences of length 10 (can be of variable length, but not focusing on that right now)
    def mini_batch_train(self, training_data, test_data, epochs, learning_rate, batch_size = 32):
        n = len(training_data)
        print("<------------------> Training Mini Batch <------------------>")

        for i in range(epochs):
            #shuffle around the sequences
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
    
    #pretty sure feedforward is correct
    #need to catch activation
    def feedforward(self, x):
        
        layer = self.layer
        a = np.zeros_like(layer.b_a) # t_0 activation is of 0s

        #x is a sequence of time points ex: [1, 2, 4, 8, 4, 2]
        for l in range(len(x)): 
            a_prev = a
            data_point = x[l]

            #fix the way we get x
            z_one = np.dot(layer.W_ax, data_point) + np.dot(layer.W_aa, a_prev) + layer.b_a
            a = layer.get_activation(z_one, 'hidden')
            
            z_two = np.dot(layer.W_ay, a) + layer.b_y
            y = layer.get_activation(z_two, 'output')

        return y
    
    def backprop(self, x, y):
        layer = self.layer

        #initializing matricies/vectors to be in the same size as the ones defined
        d_Waa = np.zeros_like(layer.W_aa)
        d_Wax = np.zeros_like(layer.W_ax)
        d_Way = np.zeros_like(layer.W_ay)
        d_Ba = np.zeros_like(layer.B_a)
        d_By = np.zeros_like(layer.B_y)

        for l in range(self.passes, 1, -1):
            d_Waa + 0

            #use cached activation for the derivatives

        return d_Waa, d_Wax, d_Way, d_Ba, d_By
    