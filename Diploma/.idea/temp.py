import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation

eps = 0.1
M = 200
N = 50
u_left = 0
u_right = 0
a = 0
b = np.pi
T = 1.
t_0 = 0.
h = (b - a) / N
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)


def tmp_f(x, t):
    tmp_var = -eps * (1 - 2 * t) * np.sin(x) \
              + 2 * np.sin(x) \
              + ((1 - 2 * t) ** 2) * np.sin(x) * np.cos(x) \
              - np.sin(3 * x) * (1 - 2 * t) * np.sin(x)
    return tmp_var

def q(x):
    return np.sin(3 * x)

def u_init(x):
    return np.sin(x)

def u_model(x,t):
    return (1-2*t)*np.sin(x)

def func(y, t, x):
    f = np.zeros((N - 1, 1))
    f.itemset(0,
              (eps * (y[1] - 2 * y[0] + u_left) / h ** 2)
              - (y[0] * (y[1] - u_left) / (2 * h))
              - y[0] * q(x[1])
              - tmp_f(x[1], t))
    for n in range(1, N - 2):
        f.itemset(n, (eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2)
                  - (y[n] * (y[n + 1] - y[n-1]) / (2 * h))
                  - y[n] * q(x[n + 1])
                  - tmp_f(x[n + 1], t))
    f.itemset(N - 2,
              (eps * (u_right - 2 * y[N - 2] + y[N - 3]) / h ** 2)
              - (y[N - 2] * (u_right - y[N - 3]) / (2 * h))
              - y[N - 2] * q(x[N - 1])
              - tmp_f(x[N - 1], t))
    return f


def func_y(y):
    f_y = np.zeros((N - 1, N - 1), dtype='float64')
    f_y.itemset((0, 0), -2 * eps / h ** 2 - (y[1] - u_left) / (2 * h) - q(x[1]))
    for n in range(1, N - 1):
        f_y.itemset((n, n - 1), eps / h ** 2 + y[n] / (2 * h))
    for n in range(1, N - 2):
        f_y.itemset((n, n), -2 * eps / h ** 2 - (y[n + 1] - y[n - 1]) / (2 * h) - q(x[n + 1]))
    for n in range(0, N - 2):
        f_y.itemset((n, n + 1), eps / h ** 2 - y[n] / (2 * h))
    f_y.itemset((-1, -1), -2 * eps / h ** 2 - (u_right - y[N - 3]) / (2 * h) - q(x[N - 1]))
    return f_y


u = np.zeros((M + 1, N + 1))
y = np.zeros((M + 1, N - 1))
for n in range(N + 1):
    u[0, n] = u_init(x[n])
y[0, :] = u[0, 1:N]

for m in range(M):
    tmp = ((1 + 1j) * (t[m + 1] - t[m]) / 2)
    tmp1 = np.eye(N - 1) - tmp * (func_y(u[m, :]))
    w_1 = np.dot(inv(tmp1), func(y[m, :], (t[m] + t[m + 1]) / 2, x)).real
    tmp2 = (t[m + 1] - t[m]) * w_1
    y[m + 1] = y[m] + np.transpose(tmp2)
    u[m + 1, 0] = u_left
    u[m + 1, 1:N] = y[m + 1]
    u[m + 1, N] = u_right

fig2 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(0, np.pi), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=1, color='red')
line2, = ax.plot([], [], lw=1, color='green')


def animate(i):
    line.set_xdata(x)
    line.set_ydata(u[i, :])
    line2.set_xdata(x)
    line2.set_ydata(u_model(x,t[i]))
    return line, line2

anim = animation.FuncAnimation(fig2, animate, frames=N+M, interval=50)
plt.show()