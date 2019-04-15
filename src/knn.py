from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


def knn__init__():
  mnist_X = np.loadtxt('datasets/MNIST_179_digits.txt')
  mnist_y = np.loadtxt('datasets/MNIST_179_labels.txt')
  
  possible_ks = []

  for k in range(1, 15):
    if k % 2 != 0:
      possible_ks.append(k)

  cv_scores = []
  
  for k in possible_ks:
    knn_clf_2 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_clf_2, mnist_X, mnist_y, cv=5, scoring='accuracy')
    
    cv_scores.append(scores.mean())
  
  mse = [x for x in cv_scores]

  optimal_k = possible_ks[mse.index(max(mse))]
  print("The optimal number of neighbors is %d" % optimal_k)

  # plot misclassification error vs k
  plt.plot(possible_ks, mse)
  plt.xlabel('Number of Neighbors K')
  plt.ylabel('Accuracy')
  plt.tight_layout()
  plt.show()
  
  print('cv_scores', cv_scores)

  # knn_clf.fit(X_train, y_train)

  # print(knn_clf.score(X_test, y_test))

knn__init__()
