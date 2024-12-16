import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf # Recommended by ChatGPT
from sklearn.preprocessing import StandardScaler
import dataHandling as dh

# Data Loading
input_data, input_labels, test_data, test_labels = dh.load_data(database, 0)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Χρησιμοποίηση μόνο 2 κατηγοριών για απλό SVM (π.χ. κατηγορίες 0 και 1)
classes = [0, 1]
train_filter = np.isin(y_train, classes).flatten()
test_filter = np.isin(y_test, classes).flatten()

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# Grayscale and normalization
x_train = np.mean(x_train, axis=3) / 255.0
x_test = np.mean(x_test, axis=3) / 255.0

# Επίπεδη απεικόνιση
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Κανονικοποίηση δεδομένων
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Εκπαίδευση SVM με γραμμικό πυρήνα
model = svm.SVC(kernel='linear', C=1.0)
model.fit(x_train, y_train.ravel())

# Αξιολόγηση
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {accuracy_train:.2f}")
print(f"Testing Accuracy: {accuracy_test:.2f}")
