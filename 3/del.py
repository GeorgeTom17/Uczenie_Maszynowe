import numpy
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("fires_thefts.csv", names=["x", "y"])
X = data["x"].to_numpy()
Y = data["y"].to_numpy() #zbieranie danych
plt.plot(X, Y, 'ro')
#plt.show()
data.to_numpy()
print(data.shape)
theta = numpy.zeros((2, 1))
newx = numpy.delete(data, numpy.s_[1:], 1)
X = numpy.insert(newx, 0, 1, axis=1)

alpha = 0.01
slope = 0
offset = 0
iter = 1500
theta1 = numpy.array([[-1], [2]])
m = len(data)
y = numpy.delete(data, numpy.s_[:1], 1)
nextout = 1/(2*m)*numpy.sum(numpy.power(numpy.subtract(numpy.dot(X, theta1), y), 2))

J_his = numpy.zeros(iter)
th_his = numpy.zeros((iter, 2))
th2 = numpy.zeros((2, 1))
for i in range(iter):
    pred = numpy.dot(x, theta1)
    th2 = th2 - (1/m)*alpha*(x.T.dot((pred - y)))
    th_his[i, :] = th2.T
    currcost = 1/(2*m)*numpy.sum(numpy.power(numpy.subtract(numpy.dot(X, th2), y), 2))
    J_his[i] = currcost
print(J_his)