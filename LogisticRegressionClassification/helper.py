from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

def get_images(mypath):

    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()
    images = np.empty(len(onlyfiles), dtype=object)
    images = np.zeros((len(onlyfiles),256,256, 3))

    for n in range(len(onlyfiles)):
        images[n] = cv2.imread( join(mypath,onlyfiles[n]))

    return images

def get_labels(array):
    labels = []
    for i in array:
        if np.max(i) > 0:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def initialize_zeros(dim):

    w = np.zeros([dim, 1])
    b = 0.0

    return w, b

def propagate(w, b, X, Y):

    #Forward
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1./m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    #Backward
    dw = (1./m) * np.dot(X, (A - Y).T)
    db = (1./m) * np.sum(A - Y)

    return dw, db, cost

def optimize(w, b, X, Y, num_iter, learning_rate):

    for i in range(num_iter):

        dw, db, cost = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            print("Iteration: {} Cost: {}".format(i, cost))

    return w, b, dw, db, cost

def predict(w, b, X):

    m = X.shape[1]
    y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0,i] >= 0.5:
            y_pred[0,i] = 1
        else:
            y_pred[0,i] = 0

    return y_pred
