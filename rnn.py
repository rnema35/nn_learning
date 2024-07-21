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
        elif activation == tanh:
            return tanh(z)
        else:
            print('we should not be here!')
            return 1
        
    def get_activation_deriv(self, z, activation_for):
        if activation_for == 'hidden':
            activation = self.input_activation
        elif activation_for == 'output':
            activation = self.output_activation

        if activation == 'sigmoid':
            return sigmoid_derivative(z)
        elif activation == 'relu':
            return relu_derivative(z)
        elif activation == tanh:
            return tanh_derivative(z)
        elif activation == 'linear':
            return linear_derivative(z)
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

            loss = 0
            for mini_batch in mini_batches:
                loss += self.update_mini_batch_params(mini_batch, learning_rate)

            accuracy = self.compute_accuracy(training_data)
            print(f"Epoch {i + 1}/{epochs}, training loss: {loss: .4f}, accuracy: {accuracy: .4f}")

        print("<------------------> Test <------------------>")
        test_loss = self.quad_cost_loss(test_data)
        test_acc = self.compute_accuracy(test_data)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    

    def update_mini_batch(self, mini_batch, learning_rate):
        layer = self.layer

        sum_Waa = np.zeros_like(layer.W_aa)
        sum_Wax = np.zeros_like(layer.W_ax)
        sum_Way = np.zeros_like(layer.W_ay)
        sum_Ba = np.zeros_like(layer.B_a)
        sum_By = np.zeros_like(layer.B_y)

        batch_loss = 0
        for x, y in mini_batch: #x and y will be arrays
            y_hypo, loss = self.feedforward(x)
            batch_loss += loss

            d_Waa, d_Wax, d_Way, d_Ba, d_By = self.backprop(x, y)

            sum_Waa += d_Waa
            sum_Wax += d_Wax
            sum_Way += d_Way
            sum_Ba += d_Ba
            sum_By += d_By

        self.W_ax -= sum_Wax
        self.W_aa -= sum_Waa 
        self.W_ay -= sum_Way 
        self.b_a -= sum_Ba 
        self.b_y -= sum_By

        return batch_loss
    
    #pretty sure feedforward is correct
    #need to cache activation
    def feedforward(self, x, y):
        
        layer = self.layer
        a = np.zeros_like(layer.b_a) # t_0 activation is of 0s

        loss = 0

        #x is a sequence of time points ex: [1, 2, 4, 8, 4, 2]
        for l in range(len(x)): 
            a_prev = a
            #get the l-th data point in the sequence
            data_point = x[l]

            z_one = np.dot(layer.W_ax, data_point) + np.dot(layer.W_aa, a_prev) + layer.b_a
            a = layer.get_activation(z_one, 'hidden')
            
            z_two = np.dot(layer.W_ay, a) + layer.b_y
            y_pred = layer.get_activation(z_two, 'output')

            # for layer l, save the output y
            self.cache['a_prev_'+str(l)] = a_prev
            self.cache['x_'+str(l)] = data_point
            self.cache['z_one_'+str(l)] = z_one
            self.cache['a_'+str(l)] = a
            self.cache['z_two_'+str(l)] = z_two
            self.cache['y_pred_'+str(l)] = y_pred

            loss += self.quad_cost_loss(y[l], y_pred)

        return y_pred, loss
    
    def backprop(self, x, y):
        layer = self.layer

        #initializing matricies/vectors to be in the same size as the ones defined
        d_Waa = np.zeros_like(layer.W_aa)
        d_Wax = np.zeros_like(layer.W_ax)
        d_Way = np.zeros_like(layer.W_ay)
        d_Ba = np.zeros_like(layer.B_a)
        d_By = np.zeros_like(layer.B_y)

        #fix these matrix multipications
        for l in range(self.passes, 1, -1):
            d_l_one = self.cost_deriv(self.cache['y_pred_'+str(l)], y) * layer.get_activation_deriv(self.cache['z_two_'+str(l)], 'output') #need to make this into np.dot
            d_By = d_l_one
            d_Way = d_l_one * self.cache['a_'+str(l)] # need to make this into a matric mulitpication
            

            d_l_two = d_l_one * layer.W_ay * layer.get_activation_deriv(self.cache['z_one_'+str(l)], 'hidden')
            #d_Waa = ____ * layer.Waa * derivative_of_g1(z_one) * a_t_prev
            d_Ba = d_l_two
            d_Waa = d_l_two * self.cache['a_prev_'+str(l)]
            d_Wax = d_l_two * self.cache['x_'+str(l)]
            
        return d_Waa, d_Wax, d_Way, d_Ba, d_By
    
    def quad_cost_loss(self, x, y):
        loss += np.sum((x - y) ** 2) / 2
        return loss

    def quad_cost_deriv(self, x, y):
        loss_deriv = x - y
        return loss_deriv
    
    def compute_accuract(self, test_data):
        correct = 0
        total = len(test_data)

        for x, y in test_data: 
            y_pred = self.feedforward(y)
            if np.isclose(y_pred, y, atol= 1e-9):
                    correct += 1

        return correct / total