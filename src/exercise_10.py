from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plotify import Plotify
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

plotify = Plotify()

plt.style.use('ggplot')

def pca(X, show_pc_plots=False, with_std=False):
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
  eigen_values, eigen_vectors, mean = pca(data, show_pc_plots=False)
  datamatrix = np.dot(np.array(eigen_vectors).T, data.T)

  return datamatrix[:d]

def kmneans__init__(n_components, show_pc_plots):
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
    print('There is ', round(len(cluster_1s) / len(cluster) * 100, 3), '% of 1s in cluster', i, 'with ', n_components, 'principal components')
    print('There is ', round(len(cluster_7s) / len(cluster) * 100, 3), '% of 7s in cluster', i, 'with ', n_components, 'principal components')
    print('There is ', round(len(cluster_9s) / len(cluster) * 100, 3), '% of 9s in cluster', i, 'with ', n_components, 'principal components')
    print('\n')


def knn__init__(n_components, show_pc_plots=False):
  mnist_X = np.loadtxt('datasets/MNIST_179_digits.txt')
  mnist_y = np.loadtxt('datasets/MNIST_179_labels.txt')

  # eigenvalues, unit_eigenvectors, mean = pca(mnist_X)
  pca = PCA(n_components=n_components)

  mnist_X = pca.fit_transform(mnist_X)
  # mnist_X = multidimension_scaling(mnist_X, n_components, show_pc_plots=show_pc_plots) #.T

  possible_ks = []

  for k in range(1, 25):
    if k % 2 != 0:
      possible_ks.append(k)
  
  cv_scores = []

  for k in possible_ks:
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_clf, mnist_X, mnist_y, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())

  mse = [x for x in cv_scores]

  optimal_k = possible_ks[mse.index(max(mse))]
  print("The optimal number of neighbors is %d" % optimal_k)

  plt.plot(possible_ks, mse)
  plt.xlabel('Number of Neighbors K')
  plt.ylabel('Accuracy')
  plt.tight_layout()
  plt.show()

  knn_clf_optimal = KNeighborsClassifier(n_neighbors=1)
  X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)
  knn_clf_optimal.fit(X_train, y_train)
  print('knn accuracy with best k = 1 is: ', knn_clf_optimal.score(X_test, y_test))


kmneans__init__(20, show_pc_plots=True)
kmneans__init__(200, show_pc_plots=True)

knn__init__(20, show_pc_plots=False)
knn__init__(200, show_pc_plots=False)

