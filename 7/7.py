import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math


data = pd.read_csv(
    "communities.data",
    delimiter=",",
    na_values=["?"]
)

temp = data.dropna(axis='rows')
final = temp.dropna(axis='columns')

data_train, data_test = train_test_split(final, test_size=0.1) #podział

m, n_plus_1 = data_train.values.shape
n = n_plus_1 - 1
Xn_train = data_train.values[:, 1:].reshape(m, n)
X_temp = np.asarray(np.concatenate((np.ones((m, 1)), Xn_train), axis=1)).reshape(m, n_plus_1)
y_train = np.asarray(data_train.values[:, 0]).reshape(m, 1)
X_train = np.delete(X_temp, 3, 1)
X_scaled_train = X_train / np.amax(X_train, axis=0)
X_normalized_train = (X_train - np.mean(X_train, axis=0)) / np.amax(X_train, axis=0)

#--------------------------------------------------------------------------

m, n_plus_1 = data_test.values.shape
n = n_plus_1 - 1
Xn_test = data_test.values[:, 1:].reshape(m, n)
X_temp_test = np.asarray(np.concatenate((np.ones((m, 1)), Xn_test), axis=1)).reshape(m, n_plus_1)
y_test = np.asarray(data_test.values[:, 0]).reshape(m, 1)
X_test = np.delete(X_temp_test, 3, 1)
X_scaled_test = X_test / np.amax(X_test, axis=0)
X_normalized_test = (X_test - np.mean(X_test, axis=0)) / np.amax(X_test, axis=0)


model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)


# The mean squared error
print("Root mean squared error before regularization : " + str(math.sqrt(mean_squared_error(y_test, y_predicted))))


for i in range(1, 21):
    alpha = i*0.2
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X_train, y_train)
    ridge_predict = clf.predict(X_test)
    print("Root mean squared error with regularization (alpha = " + str(round(alpha, 1)) + ") : " + str(math.sqrt(mean_squared_error(y_test, ridge_predict))))
