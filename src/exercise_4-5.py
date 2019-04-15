from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

style.use('fivethirtyeight')

weed_crop_train = np.loadtxt('datasets/IDSWeedCropTrain.csv', delimiter=',')
weed_crop_test = np.loadtxt('datasets/IDSWeedCropTest.csv', delimiter=',')

X_train = weed_crop_train[:, :-1]
y_train = weed_crop_train[:, -1]

X_test = weed_crop_test[:, :-1]
y_test = weed_crop_test[:, -1]


clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf.fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

# kmeans = KMeans(n_clusters=2).fit(X_train_std)
# kmeans.labels_

y_pred_knn = knn_clf.predict(X_test)

print('Mean Absolute Error for K-NN:', metrics.mean_absolute_error(y_test, y_pred_knn))
print('Mean Squared Error for K-NN:', metrics.mean_squared_error(y_test, y_pred_knn))
print('Root Mean Squared Error for K-NN:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_knn)))

y_pred_random_forest = clf.predict(X_test)
y_true = y_test
accuracy = accuracy_score(y_true, y_pred_random_forest)
print('accuracy of random forest', accuracy)

