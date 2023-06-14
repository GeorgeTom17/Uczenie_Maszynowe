import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
from sklearn.preprocessing import StandardScaler

def pca(X, k):
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    n = cov_mat.shape[0]
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    matrix_w = np.hstack([eig_pairs[i][1].reshape(n, 1)
                          for i in range(k)])
    return X_std.dot(matrix_w)


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def k_means(X, k, distance=euclidean_distance):
    history = []
    Y = []

    centroids = [[random.uniform(X.min(axis=0)[f], X.max(axis=0)[f])
                  for f in range(X.shape[1])]
                 for c in range(k)]
    history.append((centroids, Y))

    while True:
        distances = [[distance(centroids[c], x) for c in range(k)] for x in X]
        Y_new = [d.index(min(d)) for d in distances]
        if Y_new == Y:
            break
        Y = Y_new
        history.append((centroids, Y))
        XY = np.asarray(np.concatenate((X, np.matrix(Y).T), axis=1))
        Xc = [XY[XY[:, 2] == c][:, :-1] for c in range(k)]
        centroids = [[Xc[c].mean(axis=0)[f] for f in range(X.shape[1])]
                     for c in range(k)]
        history.append((centroids, Y))

    result = history[-1][1]
    return result, history, distances


def plot_clusters(X, Y, k, centroids=None):
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig = plt.figure(figsize=(16 * .7, 9 * .7))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    if not Y:
        ax.scatter(X[:, 0], X[:, 1], c='gray', marker='o', s=25, label='Dane')

    X1 = [[x for x, y in zip(X[:, 0].tolist(), Y) if y == c] for c in range(k)]
    X2 = [[x for x, y in zip(X[:, 1].tolist(), Y) if y == c] for c in range(k)]

    for c in range(k):
        ax.scatter(X1[c], X2[c], c=color[c], marker='o', s=25, label='Dane')
        if centroids:
            ax.scatter([centroids[c][0]], [centroids[c][1]], c=color[c], marker='+', s=500, label='Centroid')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.margins(.05, .05)
    return fig


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


data_flats_raw = pandas.read_table('flats_for_clustering.tsv', sep='\t')
data_flats = pandas.DataFrame()
data_flats['x1'] = data_flats_raw['cena']
data_flats['x2'] = data_flats_raw['Powierzchnia w m2']
data_flats['x3'] = data_flats_raw['Liczba pięter w budynku']
data_flats['x4'] = data_flats_raw['Piętro']
data_flats['x4'] = data_flats['x4'].apply(
    lambda x: 0 if x in ['parter', 'niski parter'] else x
)
data_flats['x4'] = data_flats['x4'].apply(
    lambda x: 2 if x in ['poddasze'] else x
)
data_flats = data_flats.dropna()

X = data_flats.values
Xs = data_flats.astype(int).values[:, 0:2]

figs = []
errors = []
minerror = 10**20
error = 0
for i in range(0, 10):
    Ys, history, J = k_means(Xs, 5)
    error = 0
    for j in J:
        error += float(sum(j))
    if error < minerror:
        plt.close()
        fig = plot_clusters(Xs, Ys, 5, centroids=history[-1][0])

X_pca = pca(X, 2)
fig = plot_unlabeled_data(X_pca)


plt.show()