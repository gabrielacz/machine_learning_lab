ALL_DATASETS = [
    ('../datasets/glass.data', 1),  # all continuous
    # ('../datasets/pima-indians-diabetes.data.txt', 9),  # all continuous
    # ('../datasets/car.data', 7),  # only nominal values
]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition, datasets
from data import Data

np.random.seed(5)


def pca_for_dataset(data, labels):
    centers = [[1, 1], [-1, -1], [1, -1]]
    X = data
    y = labels

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    # y = np.choose(y, [0, 1]).astype(np.float)

    tmp = ['r' if i=='0' else 'g' for i in labels]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=tmp , cmap=plt.cm.spectral)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

def pca_2d_for_dataset(x, y):
    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    X = pca.transform(x)
    # tmp = ['r' if i=='0' else 'g' for i in y]
    plt.scatter(X[:, 0], X[:, 1], c=y )
    plt.show()

for dataset,target_colum in ALL_DATASETS:
    data = Data()
    data.load(dataset,target_colum)
    pca_2d_for_dataset(data.dataset, data.target)



