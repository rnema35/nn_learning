import numpy as np
from functions import *

#Input layer handeled?
class Layer: 
    def __init__(self, input_dim, output_dim, activation, input_layer = False):
        if not input_layer:
            self.weights = np.random.randn(output_dim, input_dim) * 0.1 # m x n matrix
            self.bias = np.zeros((output_dim, 1)) # m x 1 matrix
        else: 
            self.weights = None
            self.bias = None
        
        self.activation = activation #i'll define a custom activation to ignore inputs
        self.final_res = []

    #to compute the activation
    def get_activation(self, z):
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        elif self.activation == 'linear':
            return linear_activation(z)
        else:
            print('we should not be here!')
            return 1

    #to compute the derivative
    def get_activation_deriv(self, z):
        if self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        elif self.activation == 'relu':
            return relu_derivative(z)
        elif self.activation == 'linear':
            return linear_derivative(z)
        else:
            print('we should not be here!')
            return 0

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
            layer = self.layers[l]
            layer.weights -= ((gradiant_sums['W'+str(l)]) * (learning_rate / batch_len))
            layer.bias -= ((gradiant_sums['B'+str(l)]) * (learning_rate / batch_len))

    #update the way to get weights and biases, and how it gets activations
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
            a = layer.get_activation(z)
            if a.ndim == 1:
                a = a.reshape((a.shape[0], 1))

            self.cache['zs'+str(l)] = z
            self.cache['activations'+str(l)] = a

        return a
    
    #update how it gets activation derivatives
    def backprop(self, x, y):
        weight_gradiant = {}
        bias_gradiant = {} 

        z_L = self.cache['zs'+str(self.num_layers - 1)]
        a_L = self.cache['activations'+str(self.num_layers-1)]

        # MAYBE - update the way to get the quadratic cost loss #
        final_layer = self.layers[-1]
        delta_l = self.quad_cost_deriv(a_L, y) * final_layer.get_activation_deriv(z_L)

        bias_gradiant['B'+str(self.num_layers-1)] = delta_l
        weight_gradiant['W'+str(self.num_layers-1)] = np.dot(delta_l, self.cache['activations'+str(self.num_layers-2)].T)

        #go backwards
        for l in range(self.num_layers-2, 0, -1):
            # MAKE SURE THIS IS WORKING RIGHT #
            currlayer = self.layers[l]
            #prevLayer = self.layers[l-1]
            nextLayer = self.layers[l+1]

            z = self.cache['zs'+str(l)]
            der_z = currlayer.get_activation_deriv(z)

            a = self.cache['activations'+str(l-1)]
            w = nextLayer.weights

            delta_l = np.dot(w.T, delta_l) * der_z

            weight_gradiant['W'+str(l)] = np.dot(delta_l, a.T)
            bias_gradiant['B'+str(l)] = delta_l

        return weight_gradiant, bias_gradiant
    
    #no need to change any of these
    def quad_cost_loss(self, data):
        loss = 0
        for x, y in data:
            output = self.feedforward(x)
            #print(output, y)
            loss += np.sum((output - y) ** 2) / 2

        return loss

    def quad_cost_deriv(self, x, y):
            loss_deriv = x - y
            return loss_deriv
    
    def compute_accuracy(self, data):
        correct = 0
        total = len(data)

        last_layer = self.layers[-1]
        if last_layer.activation != 'sigmoid':
            for x, y in data:
                output = self.feedforward(x)
                if np.isclose(output, y, atol= 1e-9):
                    correct += 1
        else: #only works for binary classification, not for multiheaded classification
            for x, y in data:
                output = self.feedforward(x)
                pred = 1 if output > 0.5 else 0
                if pred == y:
                    correct += 1

        return correct / total
    
    def get_results(self, data):
        results = []

        for x, y in data:
            results.append(self.feedforward(x)[0][0])

        return results

