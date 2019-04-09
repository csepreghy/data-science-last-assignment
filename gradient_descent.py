import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from scipy.misc import derivative
from scipy.optimize import fmin
from scipy.optimize import minimize


from matplotlib import style
import math


style.use('fivethirtyeight')

def f(x):
  return math.e**(-x/2) + 10*x**2

def df_premade(x):
  return derivative(f, x)

def df(x):
  # This is derived by using the result of this function: sp.diff(f(x))
  # return 20*x - 0.5*2.71828182845905**(-x/2)
  return 20*x - 0.5*math.e**(-x/2)



def tan_plot(a, x):
  y = df(a)*(x - a) + f(a)
  plt.plot(x, y, '--k', linewidth=1.5, zorder=3, alpha=0.8)

def steps_plot(a, x, ax):
  y = df(a)*(x - a) + f(a)
  ax.scatter(a, x, zorder=2)

def gradient_descent(alpha_list, max_iterations, precision):
  for alpha in alpha_list:
    current_x = 1
    previous_step_size = 1 #
    i = 0 # initial iteration counter

    print('df(1)', df(1))

    x = sp.Symbol('x')
    y = np.linspace((current_x-2), (current_x+2), 1000)

    fig, ax = plt.subplots()
    #plt.axis('equal')
    ax.plot(y, f(y), zorder=1, linewidth=2.5)
    
    while previous_step_size > precision and i < max_iterations and previous_step_size < 100:
      prev_x = current_x  # Store current x value in prev_x
      current_x = current_x - alpha * df(prev_x)  # Grad descent
      previous_step_size = abs(current_x - prev_x)  # Change in x

      if i < 4:
        steps_plot(current_x, f(current_x), ax)# y)
        tan_plot(current_x, y)
        # h = 0.00001
        # (f(current_x + h) - f(current_x))/h

        # ax.plot(current_x, f(current_x), 'o', markersize=10)
        # plt.plot(current_x, y, '--k')
        # plt.axhline(color='black')
        # plt.axvline(color='black')

      i += 1

    final_minimum = current_x

    minimum = minimize(f, 0) # this has the same result as the final_minimum
    print('Minimum of f(x) calculated with Scipy: ', minimum.x)
    print('The local minimum occurs at', i, 'iterations and is: ', final_minimum)

    plt.show()

gradient_descent(alpha_list=[0.1, 0.01, 0.001, 0.0001], max_iterations=10000, precision=10e-10)
