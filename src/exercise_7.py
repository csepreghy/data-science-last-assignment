import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import random
from scipy.optimize import fmin_tnc
from plotify import Plotify

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

def logistic_regression__init__():
  iris2d1_train = np.loadtxt('datasets/Iris2D1_train.txt')
  iris2d1_test = np.loadtxt('datasets/Iris2D1_test.txt')

  iris2d1_train_X = iris2d1_train[:, :-1]
  iris2d1_train_y = iris2d1_train[:, -1]

  iris2d1_test_X = iris2d1_test[:, :-1]
  iris2d1_test_y = iris2d1_test[:, -1]

  iris2d2_train = np.loadtxt('datasets/Iris2D2_train.txt')
  iris2d2_test = np.loadtxt('datasets/Iris2D2_test.txt')

  iris2d2_train_X = iris2d2_train[:, :-1]
  iris2d2_train_y = iris2d2_train[:, -1]

  iris2d2_test_X = iris2d2_test[:, :-1]
  iris2d2_test_y = iris2d2_test[:, -1]

  parameters1 = logistic_regression(iris2d1_train_X, iris2d1_train_y)
  iris2d1_test_X = np.c_[np.ones((iris2d1_test_X.shape[0], 1)), iris2d1_test_X]
  iris2d1_test_y = iris2d1_test_y[:, np.newaxis]

  accuracy1 = accuracy(iris2d1_test_X, iris2d1_test_y.flatten(), parameters=parameters1)
  print('parameters1', parameters1)
  print('accuracy1', accuracy1)

  parameters2 = logistic_regression(iris2d2_train_X, iris2d2_train_y)
  iris2d2_test_X = np.c_[np.ones((iris2d2_test_X.shape[0], 1)), iris2d2_test_X]
  iris2d2_test_y = iris2d2_test_y[:, np.newaxis]

  accuracy2 = accuracy(iris2d2_test_X, iris2d2_test_y.flatten(), parameters=parameters2)
  print('parameters2', parameters2)
  print('accuracy2', accuracy2)
  

# x = np.linspace(-10, 10, num=1000)
# plt.plot(x, sigmoid(x))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of sigmoid function')

logistic_regression__init__()
