import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import pickle

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

class NN(object):
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(nn : NN, X):
    Z1 = nn.W1.dot(X) + nn.b1
    A1 = ReLU(Z1)
    Z2 = nn.W2.dot(A1) + nn.b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, A2, nn : NN, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = nn.W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(nn : NN, dW1, db1, dW2, db2, alpha):
    nn.W1 = nn.W1 - alpha * dW1
    nn.b1 = nn.b1 - alpha * db1    
    nn.W2 = nn.W2 - alpha * dW2  
    nn.b2 = nn.b2 - alpha * db2    
    return nn

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, nn : NN):

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(nn, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, nn, X, Y)
        nn = update_params(nn, dW1, db1, dW2, db2, alpha)
        
        print("Iteration: ", i)
        predictions = get_predictions(A2)
        print(get_accuracy(predictions, Y))
    return nn

def make_predictions(X, nn : NN):
    _, _, _, A2 = forward_prop(nn, X)
    predictions = get_predictions(A2)
    return predictions

def save(nn):
    print("Saving model...")
    with open('model.pkl', 'wb') as file:
        pickle.dump(nn, file)

def load() -> NN:
    print("Loading model...")
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def test_prediction(index, nn : NN):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], nn)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def init_model()-> NN:
    try:
        nn = load()
    except:
        print("No model found, creating new model...")
        nn = NN()
    return nn

if __name__ == "__main__":
    nn = gradient_descent(X_train, Y_train, 0.10, 30, init_model())
    for i in range(20):
        test_prediction(i, nn)

    save(nn)



