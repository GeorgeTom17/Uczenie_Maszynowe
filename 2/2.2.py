# s464981

import numpy as np
import matplotlib.pyplot as plt
import math

a = 9
b = 8
c = 1

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Wykres funkcji do zadania 2.2")

x = np.arange(0.0, 10.0, 0.01)


y1 = (a-4)*x**2 + (b-5)*x + (c-6)
y2 = math.e**x/(math.e**x + 1)

ax.plot(x, y1, color="blue", lw=2)
ax.plot(x, y2, color="green", lw=2)

plt.show()


