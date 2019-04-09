from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

weed_crop_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
weed_crop_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

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


kmeans = KMeans(n_clusters=2).fit(X_train_std)
kmeans.labels_

y_pred_kmeans = kmeans.predict(X_test_std)

print('Mean Absolute Error for K-Means:', metrics.mean_absolute_error(y_test, y_pred_kmeans))
print('Mean Squared Error for K-Means:', metrics.mean_squared_error(y_test, y_pred_kmeans))
print('Root Mean Squared Error for K-Means:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_kmeans)))
