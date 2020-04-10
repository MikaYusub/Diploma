import numpy as np
from utilities import Utils

class DirectProblem:
    def direct_problem(a,b,eps, M, N, u_left, u_right, t, x, q, h,tau, u_init):


        def func(y, t, x, q):
            f = np.zeros(N - 1)
            f[0] = (eps * (y[1] - 2 * y[0] + u_left) / h ** 2) + (y[0] * (y[1] - u_left) / (2 * h)) - y[0] * q[1]
            for n in range(1, N - 2):
                f[n] = (eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2) + (y[n] * (y[n + 1] - y[n - 1]) / (2 * h)) - y[n] * q[n + 1]
            f[N - 2] = (eps * (u_right - 2 * y[N - 2] + y[N - 3]) / h ** 2) + (y[N - 2] * (u_right - y[N - 3]) / (2 * h)) - y[N - 2] * q[N - 1]
            return f

        u = np.zeros((M + 1, N + 1))
        y = np.zeros((M + 1, N - 1))
        for n in range(N + 1):
            u[0, n] = u_init[n]
        y[0, :] = u[0, 1:N]

        for m in range(M):
            a_diag, b_diag, c_diag = Utils.DiagonalPreparationDirect(N, eps, tau, q, h, y[m, :], u_left, u_right)
            w_1 = Utils.TridiagonalMatrixAlgorithm(a_diag, b_diag, c_diag, func(y[m, :], (t[m] + t[m + 1]) / 2, x, q))
            tmp2 = tau * w_1.real
            y[m + 1] = y[m] + np.transpose(tmp2)
            u[m + 1, 1:N] = y[m + 1]
            u[m + 1, 0] = u_left
            u[m + 1, N] = u_right

        return u
