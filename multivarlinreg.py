import numpy as np
# input: 1) X: the independent variables (data matrix), an (N x D)-dimensional matrix, as a numpy array
#        2) y: the dependent variable, an N-dimensional vector, as a numpy array
#
# output: 1) the regression coefficients, a (D+1)-dimensional vector, as a numpy array
#
# note: remember to either expect an initial column of 1's in the input X, or to append this within your code

def multivarlinreg(X, y, alpha, n_iters):
  if len(X.shape) == 2: n_features = X.shape[1]
  if len(X.shape) == 1: n_features = 1

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

  weights, cost = gradient_descent(weights, alpha, n_iters, hypothesis, X, y, n_features)
  return weights, cost



def get_hypothesis(weights, X, n_features):
  hypothesis = np.ones((X.shape[0], 1))
  weights = weights.reshape(1, n_features+1)
  for i in range(X.shape[0]):
    hypothesis[i] = float(np.matmul(weights, X[i]))
  
  hypothesis = hypothesis.reshape(X.shape[0])
    
  return hypothesis


def gradient_descent(weights, alpha, n_iters, hypothesis, X, y, n_features):
  cost = np.ones(n_iters)
  for i in range(n_iters):
    weights[0] = weights[0] - (alpha/X.shape[0]) * sum(hypothesis - y)
    
    for j in range(1, n_features + 1):
      weights[j] = weights[j] - (alpha/X.shape[0]) * sum((hypothesis-y) * X.T[j])
    
    hypothesis = get_hypothesis(weights, X, n_features)
    cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(hypothesis - y))
  
  weights = weights.reshape(1, n_features+1)
  return weights, cost

