import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandasgui import show
np.random.seed(1)

data = pd.read_csv('final3.csv')
# Define normalize_min_max function
def normalize_min_max(data):
    data_array = data.values
    min_vals = np.min(data_array[:, 1:], axis=0)
    max_vals = np.max(data_array[:, 1:], axis=0)
    normalized_data = data_array.copy()
    normalized_data[:, 1:] = (data_array[:, 1:] - min_vals) / (max_vals - min_vals)
    return normalized_data

data = normalize_min_max(data)

data = np.array(data)
m, n = data.shape
neurons = 8
np.random.shuffle(data)
data_dev = data[2348:m].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[0:2348].T
Y_train = data_train[0]
X_train = data_train[1:n]
_, m_train = X_train.shape
def init_params():
    W1 = np.random.rand(neurons, n-1) 
    b1 = np.random.rand(neurons, 1) 
    W2 = np.random.rand(1, neurons) 
    b2 = np.random.rand(1, 1) 
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0)

def compute_cost(A, Y):
    cost = np.sum((A - Y) ** 2) / (2 * m)
    return cost

def compute_accuracy(X, Y, W1, b1, W2, b2):
    _, _, _, A = forward_prop(W1, b1, W2, b2, X)
    predictions = (A > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)*100
    print("accuracy:", round(accuracy, 2), '%')
    return accuracy

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    dZ2 = (2 / m_train) * (A2 - Y)
    dW2 = dZ2.dot(A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = dZ1.dot(X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def gradient_descent(X_train, Y_train, X_dev, Y_dev, num_iterations, learning_rate):
    W1, b1, W2, b2 = init_params()
    costs = []
    accuracies = []
    max_accuracy = 0
    max_accuracy_iteration = 0

    for i in range(num_iterations):
        Z1_train, A1_train, Z2_train, A2_train = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = backward_prop(Z1_train, A1_train, Z2_train, A2_train, W1, W2, X_train, Y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 1 == 0:
            cost = compute_cost(A2_train, Y_train)
            costs.append(cost)
            accuracy_dev = compute_accuracy(X_dev, Y_dev, W1, b1, W2, b2)
            accuracies.append(accuracy_dev)

            if accuracy_dev > max_accuracy:
                max_accuracy = accuracy_dev
                max_accuracy_iteration = i

    # Plot the cost function and accuracy graphs
    plt.figure(figsize=(12, 4))

    # Plot cost function
    plt.subplot(1, 2, 1)
    plt.plot(range(len(costs)), costs)
    plt.xlabel('Спуски')
    plt.ylabel('Стоимость')
    plt.title('Функция стоимости')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Спуски')
    plt.ylabel('Точноcть')
    plt.title('Точность max=' + str(round(max_accuracy, 2)) + '%')



    plt.tight_layout()
    plt.show()

    print("Max Accuracy:", round(max_accuracy, 2), "%")
    print("Max Accuracy Iteration:", max_accuracy_iteration)

    return W1, b1, W2, b2

learning_rate = 0.1
num_iterations = 3000

gradient_descent(X_train, Y_train, X_dev, Y_dev, num_iterations, learning_rate)
