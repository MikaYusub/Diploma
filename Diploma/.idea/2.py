import numpy as np
import matplotlib.pyplot as plt

eps = 0.1
dBdu = -1
M = 50
N = 30
u_left = 0
u_right = 0
a = 0
b = np.pi
T = 1
t_0 = 0
tau = (T - t_0) / M
h = (b - a) / N
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)


def u_init(x):
    return (((2 - x) - (x + 4) * np.exp((3 * x - 0.75) / eps)) / (1 + np.exp((3 * x - 0.75) / eps))) + 3


# Исправить индексы


def B(u, x):
    return -u


def func(y, t, x, h, N, u_left, u_right):
    f = []
    for i in range(N - 1):
        f.append(i * 0)

    f[0] = (eps * (y[1] - 2 * y[0] + u_left) / h ** 2) - y[0] * (y[1] - u_left) / (2 * h) + B(y[0], x[1])
    for n in range(1, N - 1, 1):
        f[n] = eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2 \
               - y[n] * (y[n + 1] - y[n - 1]) / (2 * h) + B(y[n], x[n + 1])

    f[N - 1] = eps * (u_right - 2 * y[N - 1] + y[N - 2]) / h ** 2 - y[N - 1] * (u_right - y[N - 2]) / (2 * h) + B(
        y[N - 1], x[N])


def func_y(y, t, x, h, N, u_left, u_right):
    f_y = np.zeros((N - 1, N - 1))
    f_y[0, 0] = eps * (-2 / h * h) - (y[2] - u_left) / (2 * h) + dBdu
    for i in range(1, N - 2):
        f_y[i, i - 1] = eps * (1 / h ** 2) + y[i] / (2 * h)
    for i in range(1, N - 3):
        f_y[i, i] = eps * (-2 / h ** 2) - (y[i + 1] - y[i - 1]) / (2 * h) + dBdu
    for i in range(N - 3):
        f_y[i, i + 1] = eps * (1 / h ** 2) - y[i] / (2 * h)
    f_y[N - 2, N - 2] = eps * (-2 / h ** 2) - (u_right - y[N - 2]) / (2 * h) + dBdu


def example3(a, b, N, M, t_0, T, u_left, u_right, u_init):
    u = np.zeros((M + 1, N + 1))
    y = np.zeros((M + 1, N - 1))
    for n in range(N + 1):
        u[0, n] = u_init(x[n])
        # u[1, :] = u_init(x)
        y[1, :] = u[1, 2:N]
    for m in range(M):
        w_1 = (np.eye(N - 1)
               - (1 + 1j) / 2 * (t[m + 1] - t[m]) * f_y(y[m, :], t[m],
                                                        x, h, N, u_left, u_right, ))
        y[m + 1, :] = y[m, :] + (t[m + 1] - t[m]) * w_1.real
        u[m + 1, 1] = u_left
        u[m + 1, 2:N] = y[m + 1, :]
        u[m + 1, N + 1] = u_right

    plt.plot(t, u, linewidth=3.0)
    plt.xlabel('t')
    plt.ylabel('u')
    plt.show()


example3(0, 1, 100, 200, 0, 0.3, 4, -8, u_init(x))
