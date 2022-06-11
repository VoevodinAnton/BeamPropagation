import numpy as np

class State:
    def __init__(self, r0 = 1.75*10**(-3), lambda_ = 0.533*10**(-6), alpha = 0.01) -> None:
        self.r0 = r0
        self.lambda_ = lambda_
        self.alpha = alpha
        self.r = np.linspace(0, r0, 100)
        self.k = 2 * np.pi / lambda_


class Functions:
    def __init__(self, state) -> None:
         self.state = state

    def chirp(self, alpha, q):
        return np.sin((self.state.k * alpha * self.state.r) ** q)

    def radial_function(self, f, m):
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
        return F