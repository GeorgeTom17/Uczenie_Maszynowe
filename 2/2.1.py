import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data2.csv", sep=",")
data_array = data.to_numpy()


x = data_array[:, 2] #kolumna trzecia
y = data_array[:, 6] #kolumna siódma

#print(max(x)) aby upewnić się, że zakresy osi są odpowiednie
#print(max(y))

plt.plot(x, y, "go")
plt.axis([0, 6, 0, 5])

plt.show()