import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation

# from numpy import *

eps = 0.1
M = 200
N = 300
u_left = -8
u_right = 4
a = 0
b = 1
T = 0.3
t_0 = 0
tau = (T - t_0) / M
h = (b - a) / N
q = []

t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)


def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


for p in x:
    q.append(2 * p - 1 + 2 * np.sin(5 * np.pi * p) + 0.35)


def u_init(x):
    return (x ** 2 - x - 2) - 6 * np.tanh((-3 * (x - 0.25)) / eps)

# np.seterr(all='warn')
def func(u):
    f = np.zeros((N - 1, 1), dtype='float64')
    f.itemset(0, (2 * eps / (x[2] - x[0])) * ((u[2] - u[1]) / (x[2] - x[1]) - (u[1] - u_left) / (x[1] - x[0])) + \
              (u[1] * (u[2] - u_left)) / (x[2] - x[0]) - (u[1] * q[1]))
    for n in range(1, N - 1):
        kk = 2 * eps / (x[n + 1] - x[n - 1])
        k1 = ((u[n + 1] - u[n]) / (x[n + 1] - x[n])) - ((u[n] - u[n - 1]) / (x[n] - x[n - 1]))
        k2 = u[n] * ((u[n + 1] - u[n - 1]) / (x[n + 1] - x[n - 1]))
        f.itemset(n, kk * k1 + k2 - u[n] * q[n])
    t1 = 2 * eps / (x[N] - x[N - 2])
    t2 = ((u_right - u[N - 1]) / (x[N] - x[N - 1]) - ((u[N - 1] - u[N - 2]) / (x[N - 1] - x[N - 2])))
    t3 = u[N - 1] * ((u_right - u[N - 2]) / (x[N] - x[N - 2])) - (u[N - 1] * q[N - 1])
    f.itemset(-1, t1 * t2 + t3)
    return f


def func_y(u):
    f_y = np.zeros((N - 1, N - 1), dtype='float64')
    f_y.itemset((0, 0), 2 * (eps / (x[2] - x[0])) * (-1 / (x[2] - x[1]) - 1 / (x[1] - x[0])) + \
                (u[2] - u_left) / (x[2] - x[0]) - q[1])

    for n in range(1, N - 1):
        f_y.itemset((n, n - 1),
                    (2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n] - x[n - 1])) - u[n] / (x[n + 1] - x[n - 1]))
    for n in range(1, N - 2):
        f_y.itemset((n, n), (2 * eps / (x[n + 1] - x[n - 1])) * (-1 / (x[n + 1] - x[n]) - 1 / (x[n] - x[n - 1])) + \
                    (u[n + 1] - u[n - 1]) / (x[n + 1] - x[n - 1]) - q[n])
    for n in range(0, N - 2):
        f_y.itemset((n, n + 1),
                    (2 * eps / (x[n + 1] - x[n - 1])) * (1 / (x[n + 1] - x[n])) + u[n] / (x[n + 1] - x[n - 1]))

    f_y.itemset((-1, -1), (2 * eps / (x[N] - x[N - 2])) * (-1 / (x[N] - x[N - 1]) - 1 / (x[N - 1] - x[N - 2])) + (
            (u_right - u[N - 2]) / (x[N] - x[N - 2])) - q[N - 1])
    return f_y


u = np.zeros((M + 1, N + 1))
for n in range(N + 1):
    u[0, n] = u_init(x[n])

for m in range(M):
    tmp = ((1 + 1j) * (t[m + 1] - t[m]) / 2)
    tmp1 = np.eye(N - 1) - tmp * (func_y(u[m, :]))
    #w_1 = np.matmul(inv(tmp1),func(u[m, :]))
    w_1 = np.dot(inv(tmp1), func(u[m, :]))
    tmp2 = (t[m + 1] - t[m]) * w_1.real
    u[m + 1, 1:N] = u[m, 1:N] + np.transpose(tmp2)
    u[m + 1, 0] = u_left
    u[m + 1, N] = u_right

fig = plt.figure(facecolor='white')
ax = plt.axes(xlim=(a, b), ylim=(-9, 6))
line, = ax.plot([], [], lw=3)  # line = объект кривой
ax.grid(True)
plt.plot(t,x);
plt.show()
