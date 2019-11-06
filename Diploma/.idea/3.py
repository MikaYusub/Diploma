import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation
import time

eps = 0.1
M = 200
N = 50
dBdu = -1
u_left = 5.
u_right = -2.
a = 0
b = 1
T = 0.3
t_0 = 0
tau = (T - t_0) / M
h = (b - a) / N
t = []
x = []
for m in range(M + 1):
    t.append(t_0 + tau * m)


for n in range(N + 1):
    x.append(a + h * n)


# def init():
#     line.set_data([], [])
#     return line,


# def animate(i):
#     xx = np.linspace(0, 2, 1000)
#     yy = np.sin(2 * np.pi * (xx - 0.01 * i))
#     line.set_data(xx, yy)
#     return line,


def u_init(i):
    e = np.exp((3 * x[i] - 0.75) / eps)
    top = (2 - x[i]) - (x[i] + 4)*e
    down = 1 + e
    result = 3 + (top/down)
    return result


def func(y):
    f = np.zeros((N-1, 1))
    f.itemset((0, 0), (eps * (y[1] - 2 * y[0] + u_left) / (h ** 2)) - y[0] * (y[1] - u_left) / (2 * h) + B(y[0]))
    for n in range(1, N - 2):
        f.itemset((n, 0), eps*(y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2 - y[n]*(y[n+1] - y[n - 1]) / (2 * h) + B(y[n]))
    f.itemset((N-2), eps*(u_right-2*y[N - 2] + y[N - 3]) / h ** 2 - y[N - 2] * (u_right - y[N - 3])/(2 * h)+B(y[N - 2]))
    return f


def func_y(y):
    f_y = np.zeros((N - 1, N - 1))
    f_y.itemset((0, 0),  eps * (-2 / h * h) - (y[1] - u_left) / (2 * h) + dBdu)
    for i in range(1, N - 2):
        f_y.itemset((i, i - 1), eps * (1 / h ** 2) + y[i] / (2 * h))
    for i in range(1, N - 3):
        f_y.itemset((i, i), eps * (-2 / h ** 2) - (y[i + 1] - y[i - 1]) / (2 * h) + dBdu)
    for i in range(N - 3):
        f_y.itemset((i, i+1), eps * (1 / h ** 2) - y[i] / (2 * h))
    f_y.itemset((N-2, N-2), eps * (-2 / h ** 2) - (u_right - y[N - 2]) / (2 * h) + dBdu)
    return f_y


def B(u):
    return -u

#
# def func(y, t):
#     f = []
#     for i in range(N-1):
#         f.append(i*0)
#
#     f[0] = (eps*(y[1] - 2*y[0]+u_left)/h**2) - y[0]*(y[1] - u_left)/(2*h)+B(y[0])
#     for n in range(1, N-1, 1):
#         f[n] = eps*(y[n+1] - 2*y[n]+y[n-1])/h**2 \
#                - y[n]*(y[n+1]-y[n-1])/(2*h) + B(y[n])
#
#     f[N-1] = eps*(u_right - 2*y[N-1]+y[N-2])/h**2 - y[N-1]*(u_right - y[N-2])/(2*h) + B(y[N-1])
#
#
# def func_y(y, t):
#     f_y = np.zeros((N-1, N-1))
#     f_y[0, 0] = eps*(-2/h*h) - (y[2] - u_left)/(2*h) + dBdu
#     for i in range(1, N-2):
#         f_y[i, i-1] = eps*(1/h**2) + y[i]/(2*h)
#     for i in range(1, N-3):
#         f_y[i, i] = eps*(-2/h**2) - (y[i+1] - y[i-1])/(2*h)+dBdu
#     for i in range(N-3):
#         f_y[i, i+1] = eps*(1/h**2) - y[i]/(2*h)
#     f_y[N-2, N-2] = eps*(-2/h**2) - (u_right - y[N-2])/(2*h) + dBdu
#


u = np.zeros((M + 1, N + 1))
y = np.zeros((M + 1, N - 1))
for j in range(N+1):
    u[0, j] = u_init(j)
y[0, :] = u[0, 1:N]


for m in range(M):
    tmp = ((1+1j)*(t[m+1]-t[m])/2)
    tmp1 = np.eye(N - 1) - tmp * (func_y(y[m, :]))
    w_1 = np.dot(inv(tmp1), func(y[m, :]))
    tmp2 = (t[m + 1] - t[m]) * w_1.real
    y[m + 1] = y[m] + np.transpose(tmp2)
    u[m+1, 0] = u_left
    u[m+1, 1:N] = y[m+1, :]
    u[m+1, N] = u_right
plt.plot(t, u, linewidth=3.0)
plt.xlabel('t')
plt.ylabel('u')
plt.show()
