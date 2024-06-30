import numpy as np
from functions import *

class Layer: 
    def __init__(self, input_dim, output_dim, activation):
        self.weights = np.random.randn(input_dim, output_dim) * 0.1 # m x n matrix
        self.bias = np.zeros((output_dim, 1)) # m x 1 matrix
        self.activation = activation

class NN: 
    def __init__(self, layers):
        self.layers = layers #pass this as an array filled with the object layer
        self.num_layers = len(layers) 
        self.cache = {} #is there a better way to handle this?

    #don't think this function needs updating at all, nothing to do with weights/biases
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

    #update gradiant_sum initialization and the weight changes
    def update_mini_batch_params(self, mini_batch, learning_rate):
        batch_len = len(mini_batch)

        gradiant_sums = {}
        for l in range(1, self.num_layers):
            layer = self.layers[l]
            gradiant_sums['W'+str(l)] = np.zeros_like(layer.weights) # m x n matrix
            gradiant_sums['B'+str(l)] = np.zeros_like(layer.bias) # n x 1 vector

        for x, y in mini_batch: 
            x_hypo = self.feedforward(x)
            weight_gradiant, bias_gradiant = self.backprop(x_hypo, y)

            for l in range(1, self.num_layers):
                gradiant_sums['W'+str(l)] += weight_gradiant['W'+str(l)]
                gradiant_sums['B'+str(l)] += bias_gradiant['B'+str(l)]
    
        for l in range(1, self.num_layers):
            layer = self.layers[1]
            layer.weights -= ((gradiant_sums['W'+str(l)]) * (learning_rate / batch_len))
            layer.bias -= ((gradiant_sums['B'+str(l)]) * (learning_rate / batch_len))

    #update the way to get weights and biases
    def feedforward(self, x):
        self.cache = {}

        a = x.reshape((-1, 1))
        self.cache['activations'+str(0)] = a

        for l in range(1, self.num_layers): 
            layer = self.layers[l]
            a_prev = a

            weights = layer.weights
            bias = layer.bias

            z = np.dot(weights, a_prev) + bias
            a = self.linear_activation(z)
            if a.ndim == 1:
                a = a.reshape((a.shape[0], 1))

            self.cache['zs'+str(l)] = z
            self.cache['activations'+str(l)] = a

        return a
    
    # NEED TO FIX THIS - IN PROGRESS #
    def backprop(self, x, y):
        weight_gradiant = {}
        bias_gradiant = {} 

        z_L = self.cache['zs'+str(self.num_layers - 1)]
        a_L = self.cache['activations'+str(self.num_layers-1)]

        # FIX THIS LINE #
        delta_l = self.quad_cost_deriv(a_L, y) * self.linear_derivative(z_L)

        bias_gradiant['B'+str(self.num_layers-1)] = delta_l
        weight_gradiant['W'+str(self.num_layers-1)] = np.dot(delta_l, self.cache['activations'+str(self.num_layers-2)].T)

        # FIX THIS LOOP #
        for l in range(self.num_layers-2, 0, -1):
            z = self.cache['zs'+str(l)]
            liner_der_z = self.linear_derivative(z)

            a = self.cache['activations'+str(l-1)]

            #FIX THIS PARAMATER FETCH #
            w = self.params['W'+str(l+1)]

            delta_l = np.dot(w.T, delta_l) * liner_der_z

            weight_gradiant['W'+str(l)] = np.dot(delta_l, a.T)
            bias_gradiant['B'+str(l)] = delta_l

        return weight_gradiant, bias_gradiant