import numpy as np
import math
import matplotlib as plt

class NN: 
    def __init__(self, layers) -> None:
        self.layers = layers
        self.num_layers = len(layers); #how many layers are in the net
        self.params = {}
        self.cache_zs = []
        self.cache_activations = []

        for l in range(1, self.num_layers): #skip the input layer of parameters
            self.params['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) #matrix m x n, where m neurons in layer l, and n are the nuerons in the layer before
            self.params['B'+str(l)] = np.zeros((layers[l], 1)) #matrix m x 1

        # for key, value in self.params.items():
        #     print(f"KEY: {key} ; VALUE and Shape: \n {value.shape} \n {value}")
    
    def train(self, training_data, test_data, epochs, learning_rate):       
        print("<------------------> Training Stochastic <------------------>")

        for i in range(epochs):
            loss = 0
            np.random.shuffle(training_data)

            for x, y in training_data:  
                output = self.feedforward(x)
                loss += self.quad_compute_loss_one(output, y) #total
                self.update_params(output=output, g_t=y, learning_rate=learning_rate)

            if test_data:
                accuracy = self.accuracy(test_data=test_data)
                print(f"Epoch {i + 1}/{epochs}, Loss: {loss}, Accuracy: {accuracy}")
            else:
                print(f"Epoch {i + 1}/{epochs}, Loss: {loss}")

    def mini_batch_train(self, training_data, test_data, epochs, learning_rate, batch_size=32):
        n = len(training_data)
        print("<------------------> Training Mini Batch <------------------>")

        for i in range(epochs):
            np.random.shuffle(training_data)

            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch_params(mini_batch, learning_rate)

            loss = self.quad_compute_loss(training_data)
            accuracy = self.accuracy(test_data=test_data)
            print(f"Epoch {i + 1}/{epochs}, Loss: {loss}, Accuracy: {accuracy}") 

    def feedforward(self, X):
        a = X.reshape((-1, 1)) #set the input to a

        #reset the caches to store activations and zs
        self.cache_activations = []
        self.cache_zs = []

        self.cache_activations.append(a) #store the input in activations

        for l in range(1, self.num_layers): #again, skip layer 0, adding the + 1 so it becomes inclusive to get the output
            a_prev = a

            weights = self.params['W'+str(l)]
            bias = self.params['B'+str(l)]

            z = np.dot(weights, a_prev) + bias

            #a = self.sigmoid(z)
            a = self.relu(z)
            if a.ndim == 1:
                a = a.reshape((a.shape[0], 1))

            # print(f"bias shape, layer {l}: {bias.shape}")
            # print(f"z shape, layer {l}: {z.shape}")
            # print(f"a shape, layer {l}: {a.shape}")

            self.cache_zs.append(z)
            self.cache_activations.append(a)

        #a is the same thing as the activations in this case -> haven't argmaxed it or anything
        return a
    
    def backprop(self, output, g_t):
        #what do i want to return, for a matrix of paramters, I want to retrun a parameter of the same size
        weight_changes = {}
        bias_changes = {}

        #the loss for the output layer, computing vector Delta_L = Gradtiant_a(Cost) * sigmoid_prime(z_L)
        #quadratic cost derivative is the graidant take with respect to a

        z_L = self.cache_zs[-1] #the out out z
        # print(f"z_L shape: {z_L.shape}")

        #activation and output are the same
        #delta_l = self.quadratic_cost_deriv(output=output, g_t=g_t) * self.get_sigmoid_derivative(z_L)
        delta_l = self.quadratic_cost_deriv(output=output, g_t=g_t) * self.get_relu_derivative(z_L)


        # print(f"delta_l for the last layer and shape: {delta_l.shape}, {delta_l}")
        # print(f"cache activations second to last layer: {self.cache_activations[-2].shape}, {self.cache_activations[-2]}")

        bias_changes['B'+str(self.num_layers-1)] = delta_l
        weight_changes['W'+str(self.num_layers-1)] = np.dot(delta_l, self.cache_activations[-2].T) #needs the activations from the previous layer

        for l in range(self.num_layers-2, 0, -1): #skip the output and go backwards to input (exclude input)
            z = self.cache_zs[l-1] #get current layers z's (-1 because it has 1 row than activations?)
            relu_prime_z = self.get_relu_derivative(z)
            sigmoid_prime_z = self.get_sigmoid_derivative(z) #for the hammand product

            a = self.cache_activations[l-1] #would also be one layer behind l
            w = self.params['W'+str(l+1)] #get one layer ahead

            # print(f"z and its shape: {z.shape} \n {z}")
            # print(f"weights and its shape: {w.shape} \n {w}")
            # print(f"delta_l and its shape: {delta_l.shape} \n {delta_l}")
            #print(f"relu prime z and its shape: {relu_prime_z.shape} \n {relu_prime_z}")

            #the delta_l would be stored from previous, would already be a layer ahead
            #delta_l = np.dot(w.T, delta_l) * sigmoid_prime_z #from equation, loss resepect to the cost/activatoins
            delta_l = np.dot(w.T, delta_l) * relu_prime_z

            bias_changes['B'+str(l)] = delta_l #i don't understand why some people are summing this!
            weight_changes['W'+str(l)] = np.dot(delta_l, a.T)
             
        return weight_changes, bias_changes 
        
    def update_params(self, output, g_t, learning_rate):
        #gradiant is what i need to judge the parameters by
        
        weight_changes, bias_changes = self.backprop(output=output, g_t=g_t)

        #print(f"weight change dict: {weight_changes}")
        #print(f"bias change dict: {bias_changes}")

        for l in range(1, self.num_layers):
            weight = self.params['W'+str(l)]
            bias = self.params['B'+str(l)]

            self.params['W'+str(l)] = weight - (learning_rate * weight_changes['W' + str(l)])
            self.params['B'+str(l)] = bias - (learning_rate * bias_changes['B' + str(l)])

    def update_mini_batch_params(self, mini_batch, learning_rate):
        gradiant_sums = {}
        for l in range(1, self.num_layers):
            gradiant_sums['W'+str(l)] = np.zeros((self.layers[l], self.layers[l-1])) #matrix m x n, where m neurons in layer l, and n are the nuerons in the layer before
            gradiant_sums['B'+str(l)] = np.zeros((self.layers[l], 1)) #matrix m x 1

        # for key, value in gradiant_sums.items():
        #      print(f"Gradiant KEY: {key} ; Shape and VALUE: \n {value.shape} \n {value}")

        for x, y in mini_batch:
            output = self.feedforward(x) #need to cache the zs and as
            gradiant_w, gradiant_b = self.backprop(output, y)

            for l in range(1, self.num_layers):
                gradiant_sums['W'+str(l)] += gradiant_w['W'+str(l)] 
                gradiant_sums['B'+str(l)] += gradiant_b['B'+str(l)]
            
        # for key, value in gradiant_sums.items():
        #     print(f"Gradiant KEY:{key}; Shape and VALUE: \n {value.shape} \n {value}")
        
        for l in range(1, self.num_layers):
            #print(f"Before KEY: {'W'+str(l)} ; Shape and VALUE: \n {self.params['W'+str(l)].shape} \n {self.params['W'+str(l)] }")
            #print(f"Before KEY: {'B'+str(l)} ; Shape and VALUE: \n {self.params['B'+str(l)].shape} \n {self.params['B'+str(l)] }")
            self.params['W'+str(l)] -= ((learning_rate / len(mini_batch)) * gradiant_sums['W'+str(l)])  
            self.params['B'+str(l)] -= ((learning_rate / len(mini_batch)) * gradiant_sums['B'+str(l)])
            #print(f"After KEY: {'W'+str(l)} ; Shape and VALUE: \n {self.params['W'+str(l)].shape} \n {self.params['W'+str(l)] }")
            #print(f"After KEY: {'B'+str(l)} ; Shape and VALUE: \n {self.params['B'+str(l)].shape} \n {self.params['B'+str(l)] }")

    def quad_compute_loss(self, data):
        loss = 0
        for x, y in data:
            output = self.feedforward(x)
            loss += np.sum((output - y) ** 2) / 2

        return loss

    def quad_compute_loss_one(self, output, gt):
        loss = np.sum((output - gt) ** 2) / 2
        return loss

    def accuracy(self, test_data):
        # results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        # return sum(int(x==y) for (x,y) in results)

        correct = 0
        total = len(test_data)

        for x, y in test_data:
            output = self.feedforward(x)
            correct += int(output == y)

        return correct / total
    
    def quadratic_cost_deriv(self, output, g_t): #outpu = a_L ; Ground_Truth = y ; both will be vectors of the same size n x 1
        #cost function = 1/2 * (abs(y - a))^2
        #cost derivative = abs(y - a)? or no abs

        loss_deriv = output - g_t
        return loss_deriv

    def sigmoid(self, z):
        #print(f"z getting sigmoided shape: {z.shape}")
        output = 1.0 / (1.0 + np.exp(-z))  
        #print(f"z that got sigmoided shape: {output.shape}")
        return output

    def get_sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return (1-sig)*sig
        
    def relu(self, z):
        return np.maximum(0, z)
    
    def get_relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def giveHypothesis(self, test_data):
        full_output = []

        for x, _ in test_data:
            hypo_x = self.feedforward(x)
            hypo_x_flattened = hypo_x.flatten()  # Flatten if necessary
            full_output.append(hypo_x_flattened)

        return np.array(full_output)    
######################################################################################################################################################