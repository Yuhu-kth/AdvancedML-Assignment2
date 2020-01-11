"""[summary]

:return: [description]
:rtype: [type]
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
kernel = ['linear','rbf','poly']
gamma = 15
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

if __name__ == "__main__":

    Digits()
    Iris()
