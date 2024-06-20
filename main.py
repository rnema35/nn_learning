#from nn import *
from nn2 import *

X_train = np.random.randn(1000, 2)
y_train = np.array([4*a + b for a, b in X_train])

X_test = np.random.randn(1000, 2)
y_test = np.array([4*a + b for a, b in X_test])

training_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

net = NN([2, 3, 1])
#net.train(training_data= training_data, test_data= test_data, epochs=10, learning_rate=0.1)
#train_data = net.giveHypothesis(test_data=test_data)
net.mini_batch_train(training_data= training_data, test_data= test_data, epochs=20, learning_rate=0.1)
#train_mini_data = net.giveHypothesis(test_data=test_data)

#----------------------------------------------------------------------------------------------------------#
