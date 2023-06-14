import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
import seaborn

def plot_unlabeled_data(X, col1=0, col2=1, x1label=r'$x_1$', x2label=r'$x_2$'):
    fig = plt.figure(figsize=(16*.7, 9*.7))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    X1 = X[:, col1].tolist()
    X2 = X[:, col2].tolist()
    ax.scatter(X1, X2, c='k', marker='o', s=50, label='Dane')
    ax.set_xlabel(x1label)
    ax.set_ylabel(x2label)
    ax.margins(.05, .05)
    return fig


dropdown_arg1 = widgets.Dropdown(options=[0, 1, 2, 3], value=2, description='arg1')
dropdown_arg2 = widgets.Dropdown(options=[0, 1, 2, 3], value=3, description='arg2')

def interactive_unlabeled_data(arg1, arg2):
    fig = plot_unlabeled_data(
        X, col1=arg1, col2=arg2, x1label='$x_{}$'.format(arg1), x2label='$x_{}$'.format(arg2))


data_iris_raw = pandas.read_csv('iris.csv')
data_iris = pandas.DataFrame()
data_iris['x1'] = data_iris_raw['sl']
data_iris['x2'] = data_iris_raw['sw']
data_iris['x3'] = data_iris_raw['pl']
data_iris['x4'] = data_iris_raw['sw']

# Nie używamy w ogóle kolumny ostatniej kolumny ("Gatunek"),
# ponieważ chcemy dokonać uczenia nienadzorowanego.
# Przyjmujemy, że w ogóle nie dysponujemy danymi na temat gatunku,
# mamy tylko 150 nieznanych roślin.

X = data_iris.values
Xs = data_iris.values[:, 2:4]

widgets.interact(interactive_unlabeled_data, arg1=dropdown_arg1, arg2=dropdown_arg2)
seaborn.pairplot(data_iris, vars=data_iris.columns, size=1.5, aspect=1.75)
plt.show()