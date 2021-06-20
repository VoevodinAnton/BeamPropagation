import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special, integrate
import mpmath as mp


def gauss1D(rho, sigma):
    f = np.exp(-np.array(rho) ** 2 / sigma ** 2)
    return f


def gauss1Dp(rho, sigma):
    return np.exp(-rho ** 2 / sigma ** 2)


def gauss2D(x, y, sigma):
    if len(x) != len(y):
        print("x и y не равны")
    f = np.exp(-(np.array(x) ** 2 + np.array(y) ** 2) / sigma ** 2)
    return f


def radial_function(f, m):
    n = len(f) - 1
    i = np.arange(0, 2 * n + 1)
    j = np.arange(0, 2 * n + 1)
    I, J = np.meshgrid(i, j)
    index = np.around(np.sqrt((np.array(I - n) ** 2 + np.array(J - n) ** 2)))
    index[index > n] = -1
    F = np.zeros((2 * n + 1, 2 * n + 1))
    phi = np.exp(1j * m * np.arctan2(I - n, J - n))
    for n_ in i:
        for m_ in j:
            if index[n_][m_] == -1:
                F[n_][m_] = 0
                continue
            F[n_][m_] = f[int(index[n_][m_])]
    return F * phi


def plot_radial_function(F):
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(F))
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(F))
    plt.show()


def constants():
    lambda_ = 633 * 10 ** (-9)
    k = 2 * np.pi / lambda_
    alpha = 0.0628 / lambda_
    return k, alpha


# def fractional_fft_polar(f, r, m):
#     h = r[1] - r[0]
#     rho = r
#     k, alpha = constants()
#     z = 3
#     p = alpha * z
#     term1 = 1j * k * alpha / np.sin(p)
#     term2 = np.exp(1j * k * z)
#     term3 = np.exp(1j * k * alpha / (2 * np.tan(p)) * np.array(r) ** 2)
#     term4 = 1j ** (-m)
#     ker = np.exp(1j * k * alpha / (2 * np.tan(p)) * np.array(rho) ** 2) * special.jv(m, k * alpha * rho * r / np.sin(p)) * rho
#     integral = ker * f.transpose() * h
# return term2 * radial_function(term1 * term3 * term4 * integral, m)

def ker(rho, sigma, r, m, p):
    k, alpha = constants()
    return gauss1Dp(rho, sigma) * np.exp(
        1j * k * alpha / (2 * np.tan(p)) * (rho ** 2)) * special.jv(m,
                                                                    k * alpha * rho * r / np.sin(
                                                                        p)) * rho


def function_integral(sigma, r, m, p):
    return integrate.quad(ker, 0, 0.001, args=(sigma, r, m, p))[0]


def fractional_fft_polar(r, phi, z, m, sigma):
    k, alpha = constants()
    p = alpha * z
    term1 = 1j * k * alpha / np.sin(p)
    term2 = np.exp(1j * k * z)
    term3 = np.exp(1j * k * alpha / (2 * np.tan(p)) * r ** 2)
    term4 = 1j ** (-m)
    term5 = np.exp(1j * m * phi)
    integral_vec = np.vectorize(function_integral)
    g = np.exp(-r ** 2 / sigma ** 2)
    return term5 * term4 * term3 * term2 * term1 * integral_vec(sigma, r, m, p)
