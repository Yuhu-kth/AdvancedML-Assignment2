import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def data_generate(n):
    data = np.random.normal(0, 1, n)
    return data


def compute_expectation(a_N, b_N, miu_N, tau_N):
    e_miu = miu_N
    e_miuSquare = (1 / tau_N) + (miu_N ** 2)
    e_tau = a_N / b_N
    return e_miu, e_miuSquare, e_tau


def post_approximation(miu_N, tau_N, miu, tau, a_N, b_N):
    q_miu = norm(miu_N, 1 / tau_N).pdf(miu)
    q_tau = stats.gamma.pdf(tau, a_N, loc=0, scale=1 / b_N).T
    return q_miu * q_tau


def update(n, data, a_0, b_0, miu_0, lambda_0, b_N, miu, tau, tau_N):
    x_mean = (1 / n) * sum(data)
    miu_N = (lambda_0 * miu_0 + n * x_mean) / (lambda_0 + n)
    a_N = a_0 + (n + 1) / 2
    iter = 5
    for i in range(iter):
        e_miu, e_miuSquare, e_tau = compute_expectation(a_N, b_N, miu_N, tau_N)
        tau_N = (lambda_0 + n) * e_tau
        b_N = b_0 + 0.5 * lambda_0 * (e_miuSquare + miu_0 ** 2 - 2 * e_miu * miu_0) + 0.5 * sum(
            data ** 2 + e_miuSquare - 2 * e_miu * data)

        posterior = post_approximation(miu_N, tau_N, miu[:, None], tau[:, None], a_N, b_N)
        post_plot(miu, tau, posterior)


def post_plot(miu, tau, posterior):
    xGrid, yGrid = np.meshgrid(miu, tau)
    plt.contour(xGrid, yGrid, posterior, colors='b')
    plt.title('posterior approximation')
    plt.xlabel('$\mu$')
    plt.ylabel('tau')
    plt.show()


N = 10
data = data_generate(N)

a_0 = 3
b_0 = 5
miu_0 = 0.2
lambda_0 = 15

b_N = 3
tau_N = 15

miu = np.linspace(-1.0, 1.0, 200)
tau = np.linspace(-1.0, 1.0, 200)
update(N, data, a_0, b_0, miu_0, lambda_0, b_N, miu, tau, tau_N)
