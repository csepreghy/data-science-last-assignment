import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
import math

from multivarlinreg import multivarlinreg, get_hypothesis
from rmse import rmse
from src.logistic_regression import logistic_regression, accuracy
from src.kmeans import kmeans__init__
from src.knn import knn__init__

from plotify import Plotify

style.use('fivethirtyeight')


# Creating instances of imported classes
plotify = Plotify()
scaler = StandardScaler()

redwine_training = np.loadtxt('redwine_training.txt')
redwine_test = np.loadtxt('redwine_testing.txt')


X_train = redwine_training[:, :-1]
y_train = redwine_training[:, -1]

X_test = redwine_test[:, :-1]
y_test = redwine_test[:, -1]

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# multivarlinreg(X_train_std, y_train)
# weights, cost = multivarlinreg(X_train_std, y_train, alpha=0.005, n_iters=1500)
# print('weights', weights)

# X_test_std = np.concatenate((np.ones((X_test_std.shape[0], 1)), X_test_std), axis=1)
# predictions = get_hypothesis(weights, X_test_std, X_test_std.shape[1] - 1)


# rmse_all_features = rmse(predictions, y_test)
# print('rmse_all_features', rmse_all_features)
# plotify.plot(cost)

# X_train_fa = redwine_training[:, 0]
# y_train = redwine_training[:, -1]

# X_test_fa = redwine_test[:, 0]
# # X_test_fa_std = scaler.transform(X_test_fa)
# y_test = redwine_test[:, -1]

# weights_fa, cost_fa = multivarlinreg(X_test_fa, y_test, alpha=0.001, n_iters=500)

# # Add a column of of ones to the input as the bias / w0
# one_column = np.squeeze(np.ones((X_test_fa.shape[0], 1)))
# X_test_fa = np.vstack((one_column, X_test_fa.T)).T

# predictions = get_hypothesis(weights_fa, X_test_fa, 1)

# rmse_first_feature = rmse(predictions, y_test)
# print('rmse_first_feature', rmse_first_feature)
# print('weights', weights_fa)

# plotify.plot(cost_fa)


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



# parameters1 = logistic_regression(iris2d1_train_X, iris2d1_train_y)
# iris2d1_test_X = np.c_[np.ones((iris2d1_test_X.shape[0], 1)), iris2d1_test_X]
# iris2d1_test_y = iris2d1_test_y[:, np.newaxis]

# accuracy1 = accuracy(iris2d1_test_X, iris2d1_test_y.flatten(), parameters=parameters1)
# print('parameters1', parameters1)
# print('accuracy1', accuracy1)

# parameters2 = logistic_regression(iris2d2_train_X, iris2d2_train_y)
# iris2d2_test_X = np.c_[np.ones((iris2d2_test_X.shape[0], 1)), iris2d2_test_X]
# iris2d2_test_y = iris2d2_test_y[:, np.newaxis]

# accuracy2 = accuracy(iris2d2_test_X, iris2d2_test_y.flatten(), parameters=parameters2)
# print('parameters2', parameters2)
# print('accuracy2', accuracy2)


kmeans__init__()

# knn__init__()
