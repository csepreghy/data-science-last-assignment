# input: 1) X: the independent variables (data matrix), an (N x D)-dimensional matrix, as a numpy array
#        2) y: the dependent variable, an N-dimensional vector, as a numpy array
#
# output: 1) the regression coefficients, a (D+1)-dimensional vector, as a numpy array
#
# note: remember to either expect an initial column of 1's in the input X, or to append this within your code
def multivarlinreg(X, y):
    print('X', X)
