import numpy as np
from plotify import Plotify
from sklearn.preprocessing import StandardScaler


def multivarlinreg(X, y, alpha, n_iters):
  if len(X.shape) == 2:
    n_features = X.shape[1]
  if len(X.shape) == 1:
    n_features = 1

  # Add a column of of ones to the input as the bias / w0

  if len(X.shape) == 2:
    one_column = np.ones((X.shape[0], 1))
    X = np.concatenate((one_column, X), axis=1)

  if len(X.shape) == 1:
    one_column = np.squeeze(np.ones((X.shape[0], 1)))
    X = np.vstack((one_column, X.T)).T

  # Initializing weights with zeroes (+1 because of the added column of ones)
  weights = np.zeros(n_features+1)

  hypothesis = get_hypothesis(weights, X, n_features)

  weights, cost = gradient_descent(
      weights, alpha, n_iters, hypothesis, X, y, n_features)
  return weights, cost


def get_hypothesis(weights, X, n_features):
  hypothesis = np.ones((X.shape[0], 1))
  weights = weights.reshape(1, n_features+1)
  for i in range(X.shape[0]):
    hypothesis[i] = float(np.matmul(weights, X[i]))

  hypothesis = hypothesis.reshape(X.shape[0])

  return hypothesis


def rmse(f, t):
  n = len(f)
  return np.linalg.norm(f - t) / np.sqrt(n)


def gradient_descent(weights, alpha, n_iters, hypothesis, X, y, n_features):
  cost = np.ones(n_iters)
  for i in range(n_iters):
    weights[0] = weights[0] - (alpha/X.shape[0]) * sum(hypothesis - y)

    for j in range(1, n_features + 1):
      weights[j] = weights[j] - (alpha/X.shape[0]) * \
          sum((hypothesis-y) * X.T[j])

    hypothesis = get_hypothesis(weights, X, n_features)
    cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(hypothesis - y))

  weights = weights.reshape(1, n_features+1)
  return weights, cost


def exercise_2_init__():
  scaler = StandardScaler()

  redwine_training = np.loadtxt('datasets/redwine_training.txt')
  redwine_test = np.loadtxt('datasets/redwine_testing.txt')

  X_train = redwine_training[:, :-1]
  y_train = redwine_training[:, -1]

  X_test = redwine_test[:, :-1]
  y_test = redwine_test[:, -1]

  X_train_std = scaler.fit_transform(X_train)
  X_test_std = scaler.transform(X_test)
  X_test_std = np.concatenate(
      (np.ones((X_test_std.shape[0], 1)), X_test_std), axis=1)

  weights, cost = multivarlinreg(
      X_train_std, y_train, alpha=0.005, n_iters=1500)

  predictions = get_hypothesis(weights, X_test_std, X_test_std.shape[1] - 1)

  rmse_all_features = rmse(predictions, y_test)
  print('total for all features: ', rmse_all_features)

  X_train_fa = redwine_training[:, 0]
  y_train = redwine_training[:, -1]

  X_test_fa = redwine_test[:, 0]
  y_test = redwine_test[:, -1]

  weights_fa, cost_fa = multivarlinreg(
      X_train_fa, y_train, alpha=0.001, n_iters=500)

  # Add a column of of ones to the input as the bias / w0
  one_column = np.squeeze(np.ones((X_test_fa.shape[0], 1)))

  # X_test_fa_std = scaler.transform(X_test_fa)
  X_test_fa = np.vstack((one_column, X_test_fa.T)).T

  predictions = get_hypothesis(weights_fa, X_test_fa, 1)

  rmse_first_feature = rmse(predictions, y_test)
  print('rmse for first feature', rmse_first_feature)


exercise_2_init__()
