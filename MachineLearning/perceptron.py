# Perceptron with Iris data
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

# split into training / validation
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

from sklearn.linear_model import Perceptron

model = Perceptron()

model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, y_model)
print(score)

# perceptron predictions
print('predicted classes of X 0,50,100')
print(model.predict(X[[0,50,100],:]))

print()
print('model weights:')
print(model.coef_)
print()
print('bias / intercept weights')
print(model.intercept_)

w = model.coef_
b = model.intercept_

import numpy as np
print()
print('w.x+b for X[0,:]')
print(np.dot(w,(X[0,:]))+b)
print('w.x+b for X[50,:]')
print(np.dot(w,(X[50,:]))+b)
print('w.x+b for X[100,:]')
print(np.dot(w,(X[100,:]))+b)
