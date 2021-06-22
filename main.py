import numpy as np
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    # parameters
    lambda_ = 633 * 10 ** (-9)
    z = 250 * lambda_
    n1 = 0
    m1 = 2
    sigma1 = 3 * lambda_

    n2 = 0
    m2 = 5
    sigma2 = 2 * lambda_

    n3 = 0
    m3 = 7
    sigma3 = lambda_

    # Create the mesh in polar coordinates and compute corresponding Z.

    r = np.linspace(0, 10 * lambda_, 300)
    p = np.linspace(0, 2 * np.pi, 300)
    R, P = np.meshgrid(r, p)

    mode1 = utils.fractional_fft_polar(R, P, z, n1, m1, sigma1)
    mode2 = utils.fractional_fft_polar(R, P, z, n2, m2, sigma2)
    mode3 = utils.fractional_fft_polar(R, P, z, n3, m3, sigma3)

    Z = mode1

    # Express the mesh in the cartesian system.
    X, Y = utils.pol2cart(R, P)

    utils.build_contourf(X, Y, Z)

    # x = np.linspace(-10*lambda_, 10 * lambda_, 150)
    # y = np.linspace(-10*lambda_, 10 * lambda_, 150)
    # X1, Y1 = np.meshgrid(x, y)
    # Z0 = utils.gauss2D(X1, Y1, sigma)
    # utils.build_surf(X1, Y1, Z0)
    # Z1 = np.abs(utils.fractional_fft_cartesian(X1, Y1, z, sigma, m))
    # print(Z1)
    # utils.build_surf(X1, Y1, Z1)
