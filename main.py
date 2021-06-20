import numpy as np
import matplotlib.pyplot as plt
import gauss1D

if __name__ == '__main__':
    # x = np.arange(0, 2 * np.pi, 0.01)
    # y = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
    # xx, yy = np.meshgrid(x, y)
    # z = gauss1D.gauss2D(xx, yy, 3)
    # h = plt.contourf(x, y, z)
    # plt.colorbar()
    # xxx = gauss1D.gauss1D(x, 3)
    # F = gauss1D.radial_function(xxx, 3)
    # gauss1D.plot_radial_function(F)

    # f = gauss1D.gauss1D(x, 3)
    # phi =np.linspace(0, 2 * np.pi, 32)
    # sigma = 3
    # z = 3
    # ffft = gauss1D.fractional_fft_polar(f, x, phi, z, 2, sigma)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create the mesh in polar coordinates and compute corresponding Z.
    lambda_ = 633 * 10 ** (-9)
    r = np.linspace(0, 10*lambda_, 8)
    p = np.linspace(0, 2 * np.pi, 50)
    R, P = np.meshgrid(r, p)
    z = 5*lambda_
    m = 2
    sigma = 2*lambda_
    Z = np.angle(gauss1D.fractional_fft_polar(R, P, z, m, sigma))
    print(Z)

    # Express the mesh in the cartesian system.
    X, Y = R * np.cos(P), R * np.sin(P)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

    # Tweak the limits and add latex math labels.
    ax.set_zlim(0, 1)
    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')

    plt.show()
