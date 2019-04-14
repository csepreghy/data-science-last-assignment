from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plotify import Plotify
from sklearn.cluster import KMeans

plotify = Plotify()

plt.style.use('ggplot')

def pca(X, show_pc_plots=True, with_std=False):
  eigenvalues = []
  eigenvectors = []

  scaler = StandardScaler(with_std=with_std, with_mean=True)

  X = scaler.fit_transform(X)

  mean = np.mean(X, axis=0)

  covariance_matrix = np.cov(X.T)

  eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

  key = np.argsort(eigenvalues)[::-1][:len(eigenvalues)]
  eigenvalues, eigenvectors = eigenvalues[key], eigenvectors[:, key]

  unit_eigenvectors = []

  # create the unit vectors out of eigenvectors
  for v in eigenvectors:
    unit_eigenvectors.append(v / np.linalg.norm(v))

  pca = PCA(n_components=len(eigenvalues))
  pca.fit(X)

  total = sum(eigenvalues)

  var_exp = [(i / total)*100 for i in sorted(eigenvalues, reverse=True)]

  total_cumulative_explain_varience = 0
  cumulative_explain_variences = []

  for i, eigen_value in enumerate(sorted(eigenvalues, reverse=True)):
    percentage = eigen_value / total * 100
    total_cumulative_explain_varience += percentage
    cumulative_explain_variences.append(total_cumulative_explain_varience)

  if show_pc_plots == True:
    xticks = []
    for _ in range(len(eigenvalues)):
      xticks.append('PC ' + str(_))

    plotify.bar(
      x_list=range(len(eigenvalues)),
      y_list=var_exp,
      title='Explained Variance by PC',
      ylabel='% Variance Explained',
      xlabel='PCs in order of descending variance'
    )

    plotify.plot(
      y_list=cumulative_explain_variences,
      title='Cumulative Explained Variance',
      ylabel='% Variance Explained',
      xlabel='Number of Features',
    )

  return eigenvalues, unit_eigenvectors, mean


def display_digit(digit, labeled=True, title=""):
  plt.figure()
  fig = plt.imshow(digit.reshape(28, 28))
  fig.set_cmap('gray_r')
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  if title != "":
    plt.title("Inferred label: " + str(title))

def multidimension_scaling(data, d, show_pc_plots):
  eigen_values, eigen_vectors, mean = pca(data, show_pc_plots=show_pc_plots)
  datamatrix = np.dot(np.array(eigen_vectors).T, data.T)

  return datamatrix[:d]

def pca__init__(n_components):
  mnist_X = np.loadtxt('datasets/MNIST_179_digits.txt')
  mnist_y = np.loadtxt('datasets/MNIST_179_labels.txt')

  eigenvalues, unit_eigenvectors, mean = pca(mnist_X)
  mnist_X = multidimension_scaling(mnist_X, n_components, show_pc_plots=False)

  kmeans = KMeans(n_clusters=3).fit(mnist_X)
  labels = kmeans.labels_

  print('kmeans.cluster_centers_.shape', kmeans.cluster_centers_.shape)

  # for cluster_center in kmeans.cluster_centers_:
  #   display_digit(cluster_center)

  for i in range(3):
    cluster = mnist_y[np.where(labels == i)[0]]

    cluster_1s = np.where(cluster == 1)[0]
    cluster_7s = np.where(cluster == 7)[0]
    cluster_9s = np.where(cluster == 9)[0]
    print('There is ', round(len(cluster_1s) / len(cluster) * 100, 3), '% 1s in cluster', i)
    print('There is ', round(len(cluster_7s) / len(cluster) * 100, 3), '% 7s in cluster', i)
    print('There is ', round(len(cluster_9s) / len(cluster) * 100, 3), '% 9s in cluster', i)
    print('\n')

def knn(n_components):
  eigenvalues, unit_eigenvectors, mean = pca(mnist_X)
  mnist_X = multidimension_scaling(mnist_X, n_components, show_pc_plots=False)


pca__init__(20)
pca__init__(200)
