import numpy as np

from nn import NN as NN1
from nn2 import NN as NN2
from nn2 import Layer as Layer

#------------------------------------ Linear Activation Testing -------------------------------------------#

epochs = 20
learning_rate = 0.01
batch_size = 5

X_train = np.random.rand(1000, 2)
y_train = np.array([5*a + 2*b for a, b in X_train])
noise_train = np.random.normal(0, 0.0005, y_train.shape)
y_train_noisy = y_train + noise_train

X_test = np.random.rand(1000, 2)
y_test = np.array([5*a + 2*b for a, b in X_test])
noise_test = np.random.normal(0, 0.0005, y_test.shape)
y_test_noisy = y_test + noise_test

training_data = list(zip(X_train, y_train_noisy))
test_data = list(zip(X_test, y_test_noisy))

#-----------> Testing first NN

layer_sizes = [2, 3, 1]
nn1 = NN1(layer_sizes)
#nn1.mini_batch_train(training_data, test_data, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

#-----------> Testing Layer's NN

input = Layer(0, 2, 'input', True)
hidden_one = Layer(2, 3, 'linear', False)
output = Layer(3, 1, 'linear', False)
Layers = [input, hidden_one, output]
nn2 = NN2(layers=Layers)
#nn2.mini_batch_train(training_data=training_data, test_data=test_data, epochs=epochs, learning_rate=learning_rate, batch_size= batch_size)

#------------------------------------ Sigmoid Activation Testing -------------------------------------------#

epochs = 15
learning_rate = 0.1
batch_size = 1

X_train = np.random.rand(1000, 2)
y_train = np.array([1 if 5*a + 2*b > 5 else 0 for a, b in X_train])
noise_train = np.random.normal(0, 0.5, y_train.shape)
y_train_noisy = y_train + noise_train

X_test = np.random.rand(1000, 2)
y_test = np.array([1 if 5*a + 2*b > 5 else 0 for a, b in X_test])
noise_test = np.random.normal(0, 0.5, y_test.shape)
y_test_noisy = y_test + noise_test

training_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

input = Layer(0, 2, 'input', True)
hidden_one = Layer(2, 3, 'relu', False)
output = Layer(3, 1, 'sigmoid', False)
Layers = [input, hidden_one, output]

nn2 = NN2(layers=Layers)
nn2.mini_batch_train(training_data=training_data, test_data=test_data, epochs=epochs, learning_rate=learning_rate, batch_size= batch_size)
#----------------------------------------------------------------------------------------------------------#