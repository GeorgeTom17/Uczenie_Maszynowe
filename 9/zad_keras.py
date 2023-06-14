import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection

keras = tf.keras
lay = keras.layers

data = pd.read_csv('default of credit card clients.csv', index_col=0)
X = data.drop('dpnm', axis=1)
y = data['dpnm']


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=3)
model = keras.Sequential()
model.add(keras.layers.InputLayer(23))

model.add(keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.Dense(12, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.Dense(14, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
evaluate = model.evaluate(X_test, y_test)
print('Test Accuracy:', evaluate[1])
