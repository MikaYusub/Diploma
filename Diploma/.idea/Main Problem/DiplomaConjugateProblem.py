
import numpy as np
from utilities import Utils

class ConjugateProblem:
    def conjucate_problem(eps, M, N, t, x, q, h, u, f_obs,tau):
        def func_psi(y, u, t, q):
            f = np.zeros(N - 1)
            f[0] = (-eps * (y[1] - 2 * y[0]) / h ** 2) \
                   + (u[1] * y[1] / (2 * h)) \
                   + y[0] * q[1]
            for n in range(1, N - 2):
                f[n] = (-eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2) \
                       + (u[n + 1] * (y[n + 1] - y[n - 1]) / (2 * h)) \
                       + y[n] * q[n + 1]
            f[N - 2] = (-eps * (-2 * y[N - 2] + y[N - 3]) / h ** 2) \
                       - (u[N - 1] * y[N - 3] / (2 * h)) \
                       + y[N - 2] * q[N - 1]
            return f

        psi = np.zeros((M + 1, N + 1))
        y = np.zeros((M + 1, N - 1))

        psi[M, :] = -2 * (u[M, :] - f_obs)
        y[M, :] = psi[M, 1:N]
        for m in range(M, 0, -1):
            a_diag_conj, b_diag_conj, c_diag_conj = Utils.DiagonalPreparationConjugate(eps, N,tau, q, h, u[m, :])
            w_1 = Utils.TridiagonalMatrixAlgorithm(a_diag_conj,
                               b_diag_conj,
                               c_diag_conj,
                               func_psi(y[m, :], u[m, :], (t[m] + t[m - 1]) / 2, q))
            tmp2 = (t[m - 1] - t[m]) * w_1.real
            y[m - 1, :] = y[m, :] + np.transpose(tmp2)
            psi[m - 1, 1:N] = y[m - 1, :]
        psi[:, 0] = 0
        psi[:, N] = 0
        return psi