# ==================================================================
# Description:Kernel PCA applies to Iris(4-D) and Digits(64-D) dataset and
#             evaluated by using a nearest neighbor classifier(KNN)
# Author     : Yu Hu(hu3@kth.se)
# Date       : 12 Jan 2020
# ==================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (KNeighborsClassifier, NeighborhoodComponentsAnalysis)

kernel = ['linear', 'rbf', 'poly']
gamma = 0.01  # Kernel coefficient for rbf,poly kernels.Ignored by linear kernel.


def fetchData(url, names, target, features):
    """Fetch targets and data from dataset,the last column represents target"""
    df = pd.read_csv(url, names=names)
    targets = df.loc[:, target].values
    data = df.loc[:, features].values
    data = StandardScaler().fit_transform(data)

    return targets, data


# T: target; Y:Data
# fit_transform(): fit to data, then transform it
def Iris():
    """Iris dataset: 4 features, 3 classes of targets"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal length', 'sepal width', 'petal length', 'petal width',
             'target']  # List of column names of CSV file to use.
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    target = ['target']

    T, Y = fetchData(url, names, target, features)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'm', 'b']

    for i in range(3):
        # Reduce dimension to 2 with KernelPCA using different kernels
        kpca = KernelPCA(kernel=kernel[i], fit_inverse_transform=True, gamma=gamma)
        X_kpca = kpca.fit_transform(Y)
        plot('Iris', T, targets, colors, X_kpca, gamma, i)


def Digits():
    """Digits dataset: 64 features, 10 classes of targets"""
    digits = load_digits()  # Load Digits dataset
    T = digits.target
    Y = digits.data

    # The targets of Digits dataset
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['r', 'm', 'b', 'g', 'y', 'k', 'lawngreen', 'hotpink', 'magenta', 'crimson']

    for i in range(3):
        # Reduce dimension to 2 with KernelPCA using different kernels
        kpca = KernelPCA(kernel=kernel[i], fit_inverse_transform=True, gamma=gamma)
        X_kpca = kpca.fit_transform(Y)
        plot('Digits', T, targets, colors, X_kpca, gamma, i)


def plot(dataset, T, targets, colors, X_kpca, gamma, i):
    """plot seperately the results of KPCA applying to Iris and Digits dataset using 3 kinds of kernels"""
    plt.title("KPCA on {} Dataset using kernel : {},  gamma = {}".format(dataset, kernel[i], gamma))

    for t, c in zip(targets, colors):
        index = np.where(T == t)[0]
        plt.scatter(X_kpca[index, 0], X_kpca[index, 1], c=c)
    plt.show()


def Evaluate():
    """Compare different dimensionality reduction methods applied on the Digits dataset using KNN classifier.
    PCA  : Principal Component Analysis
    KPCA : Kernel Principal Component Analysis(An extension of PCA)
    GPLVM: Probabilistic Principal Component Analysis. TODO: Pending on GPLVM
    """
    n_neighbors = 3
    random_state = 0  # The seed used by the random number generator

    # Load Digits dataset
    digits = load_digits()  # Load Digits dataset
    T = digits.target
    Y = digits.data

    # Split dataset and targets into train/test subset
    Y_train, Y_test, T_train, T_test = \
        train_test_split(Y, T, test_size=0.5, stratify=T,
                         random_state=random_state)
    # Number of dimension
    dim = len(Y[0])

    # Number of targets
    n_targets = len(np.unique(T))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=2, random_state=random_state))
    # Reduce dimension to 2 with KPCA-linear kernel
    kpca_linear = make_pipeline(StandardScaler(),
                                KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True))
    # Reduce dimension to 2 with KPCA-RBF kernel
    kpca_rbf = make_pipeline(StandardScaler(),
                             KernelPCA(n_components=2, kernel='rbf', gamma=0.01, fit_inverse_transform=True))

    # Use a nearest neighbor classifier to evaluate the methods
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    # List of the methods to be compared
    dim_reduction_methods = [('PCA', pca), ('KPCA with linear kernel', kpca_linear),
                             ('KPCA with RBF kernel', kpca_rbf)]
    # plot figure
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()

        # Fit the model of the method
        model.fit(Y_train, T_train)

        # Fit a nearest neighbor classifier on the embedded training set
        KNN.fit(model.transform(Y_train), T_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        accuracy_KNN = KNN.score(model.transform(Y_test), T_test)

        # Embed the data set in 2 dimensions using the fitted model
        Y_embedded = model.transform(Y)

        # Plot the projected points and show the evaluation score
        plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, accuracy_KNN))
        plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1], c=T, s=30, cmap='Set1')

    plt.show()


if __name__ == "__main__":
    Digits()
    Iris()
    Evaluate()

