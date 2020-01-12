"""Kernel PCA which works on Iris and Digits dataset and
   evaluated by Neighborhood Components Analysis
:author: Yu Hu
"""


"""
==============================================================
Dimensionality Reduction with Neighborhood Components Analysis
==============================================================

Sample usage of Neighborhood Components Analysis for dimensionality reduction.

This example compares different (linear) dimensionality reduction methods
applied on the Digits data set. The data set contains images of digits from
0 to 9 with approximately 180 samples of each class. Each image is of
dimension 8x8 = 64, and is reduced to a two-dimensional data point.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_digits
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print(__doc__)
kernel = ['linear','rbf','poly']
gamma = 0.01

def fetchData(url, names, target, features):
    df = pd.read_csv(url, names=names)
    targets = df.loc[:, target].values
    data = df.loc[:, features].values
    data = StandardScaler().fit_transform(data)

    return targets, data

def Iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    target = ['target']
    T, Y = fetchData(url, names, target, features)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'm', 'b']

    for i in range(3):
        kpca = KernelPCA(kernel=kernel[i], fit_inverse_transform=True, gamma=gamma)
        X_kpca = kpca.fit_transform(Y)
        plot('Iris', T, targets, colors, X_kpca, gamma, i)

def Digits():
    digits = load_digits()
    Y = digits.data
    T = digits.target
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['r', 'm', 'b', 'g', 'y', 'k', 'lawngreen', 'hotpink', 'magenta', 'crimson']
    plt.figure()
    for i in range(3):
         kpca = KernelPCA(kernel=kernel[i], fit_inverse_transform=True, gamma=gamma)
         X_kpca = kpca.fit_transform(Y)
         plot('Digits',T,targets,colors,X_kpca,gamma,i)

def plot(dataset,T,targets,colors,X_kpca,gamma,i):
    plt.title("KPCA on {} Dataset using kernel : {},  gamma = {}".format(dataset,kernel[i],gamma))
    for t, c in zip(targets, colors):
        index = np.where(T == t)[0]
        plt.scatter(X_kpca[index, 0], X_kpca[index, 1], c=c)
    plt.show()

def evaluate():
    n_neighbors = 3
    random_state = 0

    # Load Digits dataset
    X, y = datasets.load_digits(return_X_y=True)

    # Split into train/test
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.5, stratify=y,
                         random_state=random_state)

    dim = len(X[0])
    n_classes = len(np.unique(y))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=2, random_state=random_state))

    kpca_linear = make_pipeline(StandardScaler(),
                                KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True))

    kpca_rbf = make_pipeline(StandardScaler(),
                             KernelPCA(n_components=2, kernel='rbf', gamma=0.01, fit_inverse_transform=True))

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Make a list of the methods to be compared
    dim_reduction_methods = [('PCA', pca), ('KPCA with linear kernel', kpca_linear),
                             ('KPCA with RBF kernel', kpca_rbf)]
    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

    # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
        plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              n_neighbors,
                                                              acc_knn))

    plt.show()

if __name__ == "__main__":
  Digits()
  Iris()
  evaluate()

