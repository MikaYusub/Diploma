import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation

eps = 0.1
M = 200
N = 300
u_left = 0
u_right = 0
a = 0
b = 1
T = 0.3
t_0 = 0
tau = (T - t_0) / M
h = (b - a) / N

t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)


def B(x, t):
    return -eps * (1 - 2 * t) * np.sin(x) + 2 * np.sin(x) \
           + ((1 - 2 * t) ** 2) * np.sin(x) * np.cos(x) - \
           np.sin(3 * np.pi * x) * (1 - 2 * t) * np.sin(x)


def q(x):
    return np.sin(3 * np.pi * x)


def u_func(x, t):
    return (1 - 2 * t) * np.sin(x)


def func(u, t):
    f = np.zeros((N - 1, 1), dtype='float64')
    f.itemset(0, (2 * eps / (x[2] - x[0])) * ((u[2] - u[1]) / (x[2] - x[1]) - (u[1] - u_left) / (x[1] - x[0])) + \
              (u[1] * (u[2] - u_left)) / (x[2] - x[0])
              - (u[1] * q[1])
              - B(N - 1, N - 1))
    for n in range(1, N - 1):
        kk = 2 * eps / (x[n + 1] - x[n - 1])
        k1 = ((u[n + 1] - u[n]) / (x[n + 1] - x[n])) - ((u[n] - u[n - 1]) / (x[n] - x[n - 1]))
        k2 = u[n] * ((u[n + 1] - u[n - 1]) / (x[n + 1] - x[n - 1]))
        f.itemset(n, kk * k1 + k2 - u[n] * q[n] - B(N - 1, N - 1))
    t1 = 2 * eps / (x[N] - x[N - 2])
    t2 = ((u_right - u[N - 1]) / (x[N] - x[N - 1]) - ((u[N - 1] - u[N - 2]) / (x[N - 1] - x[N - 2])))
    t3 = u[N - 1] * ((u_right - u[N - 2]) / (x[N] - x[N - 2])) - (u[N - 1] * q[N - 1])
    f.itemset(-1, t1 * t2 + t3 - B(N - 1, N - 1))
    return f


def func_y(u):
    f_y = np.zeros((N - 1, N - 1), dtype='float64')
    f_y.itemset((0, 0), 2 * (eps / (x[2] - x[0])) * (-1 / (x[2] - x[1]) - 1 / (x[1] - x[0]))
                + (u[2] - u_left) / (x[2] - x[0]) - q(1))

    for n in range(1, N - 1):
        f_y.itemset((n, n - 1),
                    (2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n] - x[n - 1])) - u[n] / (x[n + 1] - x[n - 1]))
    for n in range(1, N - 2):
        f_y.itemset((n, n), (2 * eps / (x[n + 1] - x[n - 1])) * (-1 / (x[n + 1] - x[n]) - 1 / (x[n] - x[n - 1])) + \
                    (u[n + 1] - u[n - 1]) / (x[n + 1] - x[n - 1]) - q(n))
    for n in range(0, N - 2):
        f_y.itemset((n, n + 1),
                    (2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n + 1] - x[n])) + u[n] / (x[n + 1] - x[n - 1]))

    f_y.itemset((-1, -1), (2 * eps / (x[N] - x[N - 2])) * (-1 / (x[N] - x[N - 1]) - 1 / (x[N - 1] - x[N - 2])) + (
            (u_right - u[N - 2]) / (x[N] - x[N - 2])) - q(N - 1))
    return f_y


u = np.zeros((M + 1, N + 1))
for n in range(N + 1):
    u[0, n] = u_func(n, 0)

for m in range(M):
    tmp = ((1 + 1j) * (t[m + 1] - t[m]) / 2)
    tmp1 = np.eye(N - 1) - tmp * (func_y(u[m, :]))
    w_1 = np.dot(inv(tmp1), func(u[m, :], (t[m] + t[m + 1]) / 2))
    tmp2 = (t[m + 1] - t[m]) * w_1.real
    u[m + 1, 1:N] = u[m, 1:N] + np.transpose(tmp2)
    u[m + 1, 0] = u_left
    u[m + 1, N] = u_right

fig = plt.figure(facecolor='white')
ax = plt.axes(xlim=(a, b), ylim=(-9, 6))
line, = ax.plot([], [], lw=3)  # line = объект кривой
ax.grid(True)


def redraw(i):
    line.set_data(t, u)


anim = animation.FuncAnimation(fig, redraw, frames=100, interval=50)
plt.show()
