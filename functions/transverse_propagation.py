import numpy as np
import matplotlib.pyplot as plt
import functions.functions as functions

if __name__ == '__main__':
    Rbeam = 1.75 * 10 ** (-3)
    r = np.linspace(10 ** (-5), Rbeam, 20)
    z = np.linspace(100 * 10 ** (-3), 850 * 10 ** (-3), 20)

    E = functions.transverse_propagation_fresnel(r, z)
    # print(E.shape)


    Z, R = np.meshgrid(z, r)
    print(Z.shape)
    functions.build_contourf(Z, R, E)

