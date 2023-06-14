from mpl_toolkits.mplot3d import (Axes3D)
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

X = np.arange(-10, 10, 0.15)
Y = np.arange(-10, 10, 0.15)
X, Y = np.meshgrid(X, Y)

Z = -1 * (X**2 + Y**3)

powierzchnia = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True)

plt.show()