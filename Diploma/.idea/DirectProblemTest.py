import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation

eps = 0.1
M = 100
N = 50
u_left = 0
u_right = 0
a = 0
b = np.pi
T = 1
t_0 = 0
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)


def tmp_f(x, t):
    s = -eps * (1 - 2 * t) * np.sin(x) + 2 * np.sin(x) \
           + ((1 - 2 * t) ** 2) * np.sin(x) * np.cos(x) - \
           np.sin(3 * np.pi * x) * (1 - 2 * t) * np.sin(x)

    return s

def q(x):
    return np.sin(3 * x)


def u_func(x, t):
    return (1 - 2 * t) * np.sin(x)


def func(u, t):
    f = np.zeros((N - 1, 1), dtype='float64')
    f.itemset(0, (2 * eps / (x[2] - x[0])) * ((u[2] - u[1]) / (x[2] - x[1]) - (u[1] - u_left) / (x[1] - x[0])) + \
              (u[1] * (u[2] - u_left)) / (x[2] - x[0]) - (u[1] * q(x[1])) - tmp_f(x[0], t)
              )
    for n in range(1, N - 1):
        kk = 2 * eps / (x[n + 1] - x[n - 1])
        k1 = ((u[n + 1] - u[n]) / (x[n + 1] - x[n])) - ((u[n] - u[n - 1]) / (x[n] - x[n - 1]))
        k2 = u[n] * ((u[n + 1] - u[n - 1]) / (x[n + 1] - x[n - 1]))
        f.itemset(n, kk * k1 + k2 - u[n] * q(x[n]) - tmp_f(x[n], t))
    t1 = 2 * eps / (x[N] - x[N - 2])
    t2 = ((u_right - u[N - 1]) / (x[N] - x[N - 1]) - ((u[N - 1] - u[N - 2]) / (x[N - 1] - x[N - 2])))
    t3 = u[N - 1] * ((u_right - u[N - 2]) / (x[N] - x[N - 2])) - (u[N - 1] * q(x[N - 1]))
    f.itemset(-1, t1 * t2 + t3 - tmp_f(x[N - 1], t))
    return f

def func_y(u):
    f_y = np.zeros((N - 1, N - 1), dtype='float64')
    f_y.itemset((0, 0), -2 * (eps / (x[2] - x[0])) * (1 / (x[2] - x[1]) + 1 / (x[1] - x[0])) + \
                (u[2] - u_left) / (x[2] - x[0]) - q(x[1]))
    for n in range(1, N - 1):
        f_y.itemset((n, n - 1),
                    (2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n] - x[n - 1])) - u[n] / (x[n + 1] - x[n - 1]))
    for n in range(1, N - 2):
        f_y.itemset((n, n), (-2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n + 1] - x[n]) + 1 / (x[n] - x[n - 1])) + \
                    (u[n + 1] - u[n - 1]) / (x[n + 1] - x[n - 1]) - q(x[n]))
    for n in range(0, N - 2):
        f_y.itemset((n, n + 1),
                    (2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n + 1] - x[n])) + u[n] / (x[n + 1] - x[n - 1]))
    f_y.itemset((-1, -1), (-2 * eps / (x[N] - x[N - 2])) * (1 / (x[N] - x[N - 1]) + 1 / (x[N - 1] - x[N - 2])) +
                (u_right - u[N - 2]) / (x[N] - x[N - 2]) - q(x[N - 1]))
    return f_y


u = np.zeros((M + 1, N + 1))
y = np.zeros((M + 1, N - 1))
for n in range(N + 1):
    u[0, n] = u_func(x[n], t_0)
y[0, :] = u[0, 1:N]

for m in range(M):
    tmp = ((1 + 1j) * (t[m + 1] - t[m]) / 2)
    tmp1 = np.eye(N - 1) - tmp * (func_y(u[m, :]))
    w_1 = np.dot(inv(tmp1), func(u[m, :], (t[m] + t[m + 1]) / 2))
    tmp2 = (t[m + 1] - t[m]) * w_1.real
    y[m + 1] = y[m] + np.transpose(tmp2)
    u[m + 1, 0] = u_left
    u[m + 1, 1:N] = y[m + 1]
    u[m + 1, N] = u_right

plt.axis([0, 1, -100, 100])
plt.plot(t, u);
plt.show()

fig2 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(0, 1), ylim=(-500, 500))
line, = ax.plot([], [], lw=3)

# def redraw(i):
#     x = t[i]
#     y = func(u[i, :],t[i])
# #u[:, i+1]
#     line.set_data(x, y)
# anim = animation.FuncAnimation(fig2, redraw, frames=126, interval=100)
# plt.show()
