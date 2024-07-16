import numpy as np
import matplotlib.pyplot as plt

# num_layer = 5

# for i in range (2, 5):
#     print(f"normal: {-i}")
#     print(f"normal + 1: {-i + 1}")
#     print(f"normal - 1: {-i - 1}")
    
# for i in range (5 - 2, 0, -1):
#     print(f"normal: {i}")
#     print(f"normal + 1: {i + 1}")
#     print(f"normal - 1: {i - 1}")]


#lets try with a sin function
X = np.arange(0, 7, 0.01)
y = np.sin(X)

train = []

for i in range(len(X) - 9):
    temp = np.array(y[i:i+10])
    train.append(temp)

train = np.array(train)

print(train.shape)
print(train[-1])