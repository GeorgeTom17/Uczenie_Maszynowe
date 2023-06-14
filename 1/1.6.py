import numpy as np
X = np.array([[1, 2, 3], [1, 3, 6]])
y = np.array([[5, 6]]).T
wynik = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print(wynik)