import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style

def kmeans__init__():
  style.use('fivethirtyeight')
  
  mnist_X = np.loadtxt('datasets/MNIST_179_digits.txt')
  mnist_y = np.loadtxt('datasets/MNIST_179_labels.txt')

  print('mnist_X', mnist_X.shape)
  print('mnist_y', mnist_y)

  kmeans = KMeans(n_clusters=3).fit(mnist_X)
  labels = kmeans.labels_

  for cluster_center in kmeans.cluster_centers_:
    display_digit(cluster_center)
  
  plt.show()

  for i in range(3):
    cluster = mnist_y[np.where(labels == i)[0]]

    cluster_1s = np.where(cluster == 1)[0]
    cluster_7s = np.where(cluster == 7)[0]
    cluster_9s = np.where(cluster == 9)[0]
    print('There is ', round(len(cluster_1s) / len(cluster) * 100, 3), '% 1s in cluster', i)
    print('There is ', round(len(cluster_7s) / len(cluster) * 100, 3), '% 7s in cluster', i)
    print('There is ', round(len(cluster_9s) / len(cluster) * 100, 3), '% 9s in cluster', i)
    print('\n')


def display_digit(digit, labeled=True, title=""):
  plt.figure()
  fig = plt.imshow(digit.reshape(28, 28))
  fig.set_cmap('gray_r')
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  if title != "":
    plt.title("Inferred label: " + str(title))


kmeans__init__()
