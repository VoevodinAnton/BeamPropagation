import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy import special, integrate
import sympy as sp

if __name__ == '__main__':
    # # parameters
    k, alpha = utils.constants()
    Rbeam = 1.75 * 10 ** (-3)
    q = 3
    print("q: " + str(q))
    a = 0.000142
    k, alpha = utils.constants()

    lambda_ = 2 * np.pi / k
    print("длина волны: " + str(lambda_))
    z = 500 * (10 ** (-3))
    print("z: " + '{:06.2f}'.format(z * 10 ** 3) + " mm")
    n1 = 0
    m1 = 2
    sigma1 = 3 * lambda_

    n2 = 0
    m2 = 5
    sigma2 = 2 * lambda_

    n3 = 0
    m3 = 7
    sigma3 = lambda_


    # фокус:
    zmax = 2 ** ((q - 2) / q) * Rbeam / (q * a * (k * a * Rbeam) ** (q - 1))
    print("фокус: " + '{:06.2f}'.format(zmax * 10 ** 3) + " mm")

    r = np.linspace(10 ** (-5), Rbeam, 50)
    p = np.linspace(0, 2 * np.pi, 50)
    R, P = np.meshgrid(r, p)

    # mode1 = utils.fractional_fft_polar(R, P, z, n1, m1, sigma1)
    # mode2 = utils.fractional_fft_polar(R, P, z, n2, m2, sigma2)
    # mode3 = utils.fractional_fft_polar(R, P, z, n3, m3, sigma3)

    X, Y = utils.pol2cart(R, P)

    # Initial field
    E0 = np.sin((a * k * R) ** q)
    utils.build_contourf(X, Y, E0)

    E = utils.fresnel_transform(z, R)

    utils.build_contourf(X, Y, E)

    # x = np.linspace(-10*lambda_, 10 * lambda_, 150)
    # y = np.linspace(-10*lambda_, 10 * lambda_, 150)
    # X1, Y1 = np.meshgrid(x, y)
    # Z0 = utils.gauss2D(X1, Y1, sigma1)
    # utils.build_contourf(X1, Y1, Z0)
    # Z1 = np.abs(utils.fractional_fft_cartesian(X1, Y1, z, sigma1, m1))
    # print(Z1)

    # utils.build_contourf(X1, Y1, Z1)
