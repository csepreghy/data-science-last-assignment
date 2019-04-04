import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

from multivarlinreg import multivarlinreg
from plotify import Plotify

# Creating instances of imported classes
plotify = Plotify()
scaler = StandardScaler()

redwine_training = np.loadtxt('redwine_training.txt')


X_train = redwine_training[:, :-1]
y_train = redwine_training[:, -1]

X_train_std = scaler.fit_transform(X_train)

# multivarlinreg(X_train_std, y_train)
weights, cost = multivarlinreg(X_train_std, y_train)

plt.plot(cost)
plt.show()
