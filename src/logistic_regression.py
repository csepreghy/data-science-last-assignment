import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import random
from scipy.optimize import fmin_tnc

def logistic_regression(X, y, X_test=[], y_test=[]):
  style.use('fivethirtyeight')
  plt.xlim(left=np.min(X[:,0])-0.1, right=np.max(X[:,0])+0.1)
  plt.ylim(top=np.max(X[:,1])+0.1, bottom=np.min(X[:,1])-0.1)

  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1])
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1])

  X_train = np.c_[np.ones((X.shape[0], 1)), X]
  y_train = y[:, np.newaxis]
  theta = np.zeros((X_train.shape[1], 1))

  x_values = [np.min(X_train[:, 1] - 2), np.max(X_train[:, 2] + 2)]

  parameters = fit(X_train, y_train, theta)

  y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

  plt.plot(x_values, y_values, label='Decision Boundary', color='green', linewidth=4)

  plt.xlabel('f')
  plt.ylabel('Marks in 2nd Exam')
  plt.legend()

  plt.show()

  return parameters



def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def probability(theta, x):
  return sigmoid(np.dot(x, theta))


def cost_function(theta, x, y):
  m = x.shape[0]
  total_cost = -(1 / m) * np.sum(
    y * np.log(probability(theta, x)) + (1 - y) * np.log(1 - probability(theta, x)))
  return total_cost


def gradient(theta, x, y):
  m = x.shape[0]
  return (1 / m) * np.dot(x.T, sigmoid(np.dot(x, theta)) - y)


def fit(x, y, theta):
  opt_weights = fmin_tnc(
    func=cost_function,
    x0=theta,
    fprime=gradient, 
    args=(x, y.flatten())
  )

  return opt_weights[0]


def predict(x, parameters):
    theta = parameters[:, np.newaxis]
    return probability(theta, x)


def accuracy(x, actual_classes, parameters, probab_threshold=0.5):
    predicted_classes = (predict(x, parameters) >=
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100




# x = np.linspace(-10, 10, num=1000)
# plt.plot(x, sigmoid(x))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of sigmoid function')

