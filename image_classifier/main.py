import time as t
import numpy as np
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from model_utils import load_dataset

train_set_x_orig, train_set_y, classes, labels = load_dataset()
num_px=train_set_x_orig.shape[1]

def sigmoid(z):
    s=1 / (1+np.exp(-z))
    return s

def tan_h(z):
    t=((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
    return t

br="--------------------------------------------------------"

print("Dataset loaded", br)
print("Orig Train set X.shape: "+str(train_set_x_orig.shape)+"")
print("Orig Train set Y.shape: "+str(train_set_y.shape)+"")
print("Classes: "+str(classes)+"")
print(f"Labels: {str(labels)}", br)

train_set_x=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T

train_set_x=train_set_x/255.

def initialize_params(dim,c):
    w, b = np.zeros([dim,c]), 0
    assert(w.shape == (dim,c))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def propagation(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    # The cost function
    cost = -1 / m * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1 - A).T))

    dw = (1 / m) * np.dot(X, (A-Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    grads={
        "dw":dw,
        "db":db
        }

    return grads, cost


def optimize(w,b,X,Y,iterations,learning_rate,print_cost=True):
    m = X.shape[1]
    costs = []
    for i in range(iterations):
        grads, cost = propagation(w,b,X,Y)

        dw=grads["dw"]
        db=grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after "+str(i)+"iterations: "+str(cost))
    params={
        "w":w,
        "b":b
        }
    costs=np.squeeze(costs)
    return params, grads, costs

def softmax(z):
    t = np.exp(z)
    a = t / (np.sum(t))
    return a

def predict(w,b,X):
    m = X.shape[1]
    Z = np.dot(w.T,X)+b
    A = softmax(Z)
    Y_hat = np.zeros([5,m])
    """for i in range(A.shape[1]):
        for n in range(A.shape[0]):
            if A[n][0] >= 0.5:
                print(labels[0][i])
                Y_hat[n][i][0] = 1
            else:
                print(0)
                Y_hat[i][0] = 0"""
    for i in range(m):
        for n in range(A.shape[0]):
            if A[n][0] >= 0.5:
                print(labels[0][n])
                Y_hat[n][i] = 1
            else:
                Y_hat[n][i] = 0
    return Y_hat
    print("Y hat: ")
    print(Y_hat)
    print("--------------------------------")
    print(Y_hat.shape)
    print("--------------------------------")
    print(A)
    print("--------------------------------")
    print(A.shape)
    print("--------------------------------")
    print(A[3][0])
    



def model(X_train, Y_train, learning_rate, iterations, print_cost=True):
    
    w, b = initialize_params(X_train.shape[0],classes.shape[1])
    tik = t.time()
    params, grads, costs=optimize(w, b, X_train, Y_train, iterations, learning_rate, print_cost=True)
    print(t.time()-tik)
    w = params["w"]
    b = params["b"]

    Y_train_predictions=predict(w, b, X_train)
    print("Y TRAIN HATS")
    print("------------------------------------------------------------------------------------------")
    print(Y_train_predictions)
    
    data = {
        "w":w,
        "b":b,
        "dw":grads["dw"],
        "db":grads["db"],
        "costs":costs
        }
    return data
                   

data = model(train_set_x, train_set_y, 0.0000998, 1500, print_cost=True)



def my_image(filepath):
    image = Image.open(rf"{filepath}")
    arr = np.array(image)
    arr = arr.reshape(1,-1).T
    return arr

w, b = data["w"], data["b"]
print("--------------------YOUR PREDICTION IS:------------------------------")
predict(w, b, my_image("random_image.jpg"))
