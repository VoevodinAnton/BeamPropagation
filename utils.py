import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special, integrate
import mpmath as mp


def gauss1D(rho, sigma):
    return np.exp(-rho ** 2 / sigma ** 2)


def gauss_laguerre(rho, sigma, n, m):
    t = (rho ** 2) / (sigma ** 2)
    term1 = (np.sqrt(2) * rho / sigma) ** np.abs(m)
    term2 = special.assoc_laguerre(2 * t, n, np.abs(m))
    term3 = np.exp(-t)
    return term1 * term2 * term3


def gauss_bessel(rho, sigma, m):
    k, alpha = constants()
    kr = alpha * np.sin(0.3)
    t = (rho ** 2) / (sigma ** 2)
    term1 = special.jv(m, kr * rho)
    term2 = np.exp(-t)
    return term1 * term2


def gauss2D(x, y, sigma):
    f = np.exp(-(x ** 2 + y ** 2) / sigma ** 2)
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
    lambda_ = 532 * 10 ** (-9)
    k = 2 * np.pi / lambda_
    alpha = 0.0628 / lambda_
    return k, alpha


def fractional_fft_polar_radial(f, r, m):
    h = r[1] - r[0]
    rho = r
    k, alpha = constants()
    z = 3
    p = alpha * z
    term1 = 1j * k * alpha / np.sin(p)
    term2 = np.exp(1j * k * z)
    term3 = np.exp(1j * k * alpha / (2 * np.tan(p)) * np.array(r) ** 2)
    term4 = 1j ** (-m)
    ker = np.exp(1j * k * alpha / (2 * np.tan(p)) * np.array(rho) ** 2) * special.jv(m, k * alpha * rho * r / np.sin(
        p)) * rho
    integral = ker * f.transpose() * h
    return term2 * radial_function(term1 * term3 * term4 * integral, m)


def ker_polar(rho, sigma, r, n, m, p):
    k, alpha = constants()
    a = 0.0018
    beta = k * a
    term0 = np.sin((beta * (rho - 2 * 10 ** (-3))) ** (4 / 3))
    term1 = gauss_laguerre(rho, sigma, n, m)
    term2 = np.exp(1j * k * alpha * (rho ** 2) / (2 * np.tan(p)))
    term3 = special.jv(m, k * alpha * rho * r / np.sin(p)) * rho
    return term1 * term2 * term3


def function_integral_polar(sigma, r, n, m, p):
    integral = integrate.quad(ker_polar, 0, 0.0001, args=(sigma, r, n, m, p))
    print("погрешность интегрирования:")
    print(integral[1])
    return integral[0]


def fractional_fft_polar(r, phi, z, n, m, sigma):
    k, alpha = constants()
    p = alpha * z
    term1 = -1j * k * alpha / np.sin(p)
    term2 = np.exp(1j * k * z)
    term3 = np.exp(1j * k * alpha / (2 * np.tan(p)) * r ** 2)
    term4 = 1j ** (-m)
    term5 = np.exp(1j * m * phi)
    integral_vec = np.vectorize(function_integral_polar)
    g = np.exp(-r ** 2 / sigma ** 2)
    return term5 * term4 * term3 * term2 * term1 * integral_vec(sigma, r, n, m, p)


def pol2cart(r, p):
    x, y = r * np.cos(p), r * np.sin(p)
    return x, y


def ker_cartesian(ksi, eta, x, y, p, sigma, m):
    k, alpha = constants()
    phi = np.arctan2(eta, ksi)
    term1 = gauss2D(ksi, eta, sigma) * np.exp(1j * m * phi)
    term2 = np.exp(1j * k * alpha * (ksi ** 2 + eta ** 2) / (2 * np.tan(p)))
    term3 = np.exp(-1j * k * alpha * (ksi * x + eta * y) / np.sin(p))
    return term1 * term2 * term3


def function_integral_cartesian(x, y, p, sigma, m):
    integral = integrate.nquad(ker_cartesian, [[0, 0.0001], [0, 0.0001]], args=(x, y, p, sigma, m))
    print("погрешность интегрирования:")
    print(integral[1])
    return integral[0]


def fractional_fft_cartesian(x, y, z, sigma, m):
    k, alpha = constants()
    p = alpha * z

    term1 = -1j * k * alpha / (2 * np.pi * np.sin(p))
    term2 = np.exp(1j * k * z)
    term3 = np.exp(1j * k * alpha * (x ** 2 + y ** 2) / (2 * np.tan(p)))

    integral_vec = np.vectorize(function_integral_cartesian)

    return term1 * term2 * term3 * integral_vec(x, y, p, sigma, m)


def ker_fresnel_polar(rho, z, r):
    k, alpha = constants()
    a = 0.000142
    q = 3
    term0 = np.sin((a * k * rho) ** q)
    # term0 = np.sin((beta * rho) ** (5 / 2))
    # term0 = np.exp(-1j * (k * a * rho) ** q)
    term2 = np.exp(1j * k * (rho ** 2) / (2 * z))
    term3 = np.exp(- 1j * k * (r * rho) / z)
    term4 = np.sqrt(rho)
    return term0 * term2 * term3 * term4


def function_integral_fresnel_polar(z, r):
    integral = integrate.quad(ker_fresnel_polar, 0, np.inf, args=(z, r))
    # print("погрешность интегрирования:")
    # print(integral[1])
    return integral[0]


def fresnel_transform(z, r):
    k, alpha = constants()
    term1 = -1j * np.exp(1j * np.pi / 4)
    term2 = np.sqrt(k / (2 * np.pi * z * r))
    term3 = np.exp(1j * k * (r ** 2) / (2 * z))
    integral_vec = np.vectorize(function_integral_fresnel_polar)
    return term1 * term2 * term3 * integral_vec(z, r)


def build_surf(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

    # Tweak the limits and add latex math labels.
    ax.set_zlim(0, 35)
    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')

    plt.show()


def build_contourf(X, Y, Z):
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, np.abs(Z))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, np.angle(Z), cmap='gist_gray')
    plt.show()
