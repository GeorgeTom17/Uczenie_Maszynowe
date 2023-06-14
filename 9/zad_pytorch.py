import sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class Net(nn.Module):
  def __init__(self, input_shape):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_shape, 25)
    self.fc2 = nn.Linear(25, 35)
    self.fc3 = nn.Linear(35, 1)
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x


data = pd.read_csv('default of credit card clients.csv', index_col=0)

print(torch.cuda.device_count())

learning_rate = 0.05
epochs = 700

X_temp = data.drop('dpnm', axis=1)
y_temp = data['dpnm']

X = torch.tensor(X_temp.values)
y = torch.tensor(y_temp.values)

sc = StandardScaler()
X = sc.fit_transform(X)

trainset = dataset(X, y)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

model = Net(input_shape=X.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

losses = []
accur = []

for i in range(epochs):
    for j, (x_train, y_train) in enumerate(trainloader):
        output = model(x_train)
        loss = loss_fn(output, y_train.reshape(-1, 1))
        predicted = model(torch.tensor(X, dtype=torch.float32))
        acc = (predicted.reshape(-1).detach().numpy().round().mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 2 == 0:
        losses.append(loss)
        accur.append(acc)
        print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))