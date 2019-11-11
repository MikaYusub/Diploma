import numpy as np
import matplotlib.pyplot as plt

eps = 0.1


def f(u, t):
    return (1 / eps) * u * (t - u)


def f_u(u, t):
    return (1 / eps) * (t - 2 * u)


def example(u_0, t_0, T, M):
    tau = (T - t_0) / M
    m: int

    t = np.linspace(t_0, T, M + 1)
    u = np.zeros(M + 1);
    u[0] = u_0

    for m in range(M):
        w_1 = f(u[m], (t[m + 1] + t[m]) / 2) / (1 - ((1 + 1j) / 2) * (t[m + 1] - t[m]) * f_u(u[m], t[m]))
        u[m + 1] = u[m] + (t[m + 1] - t[m]) * w_1.real
        print(u[m])

    plt.plot(t, u, linewidth=3.0)
    plt.axis([-1, 2, 0, 3])
    plt.xlabel('t')
    plt.ylabel('u')
    plt.show()


example( 3, -1, 2, 50)
