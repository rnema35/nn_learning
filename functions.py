import numpy as np

def linear_activation(z):
    const = 1
    return z * 1

def linear_derivative(z):
    return 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def tanh(z):
    pos_e = np.exp(z)
    neg_e = np.exp(-z)
    return ((pos_e - neg_e) / (pos_e + neg_e))

def tanh_derivative(z):
    tanh_z = tanh(z)
    return 1 - np.power(tanh_z, 2)