import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation

eps = 0.001
M = 200
N = 500
u_left = 0
u_right = 0
a = 0
b = np.pi
T = 1.
t_0 = 0.
h = (b - a) / N
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)
u = []
q = []


def tmp_f(x, t):
    tmp_var = -eps * (1 - 2 * t) * np.sin(x) \
              - 2 * np.sin(x) \
              - np.exp(-x * t) * (1 - 2 * t) * np.cos(x) \
              - 2 * np.cos(x) * (1 - 2 * t) * np.sin(x)
    return tmp_var


def q_init(x):
    return 2 * np.cos(x)


for n in range(0, N + 1):
    q.append(q_init(x[n]))


def u_init(x):
    return x


for n in range(N + 1):
    u.append(u_init(n))


def u_model(x, t):
    return x * np.exp(-x * t)


def func_psi(psi, u, t,q):
    f = np.zeros((N - 1, 1))
    f.itemset(0,
              (-eps * (psi[1] - 2 * psi[0]) / h ** 2)
              + (u[0] * psi[1] / (2 * h))
              + psi[0] * q[1]
              )
    for n in range(1, N - 2):
        f.itemset(n,
                  (-eps * (psi[n + 1] - 2 * psi[n] + psi[n - 1]) / h ** 2)
                  + (u[n] * (psi[n + 1] - psi[n - 1]) / (2 * h))
                  + psi[n] * q[n + 1])
    f.itemset(N - 2,
              (-eps * (-2 * psi[N - 2] + psi[N - 3]) / h ** 2)
              - (u[N - 2] * psi[N - 3] / (2 * h))
              + psi[N - 2] * q[N - 1])
    return f


def func_y_psi(u, q):
    f_y = np.zeros((N - 1, N - 1), dtype='float64')
    for n in range(1, N - 1):
        f_y.itemset((n, n - 1), -eps / h ** 2 - u[n] / (2 * h))
    for n in range(0, N - 1):
        f_y.itemset((n, n), 2 * eps / h ** 2 + q[n])
    for n in range(0, N - 2):
        f_y.itemset((n, n + 1), -eps / h ** 2 + u[n] / (2 * h))
    return f_y

y = np.zeros((M + 1, N + 1))
psi = np.zeros((M + 1, N - 1))
for n in range(N + 1):
    y[0, n] = u_init(x[n])
psi[0, :] = y[0, 1:N]

for m in range(1,M):
    tmp = ((1 + 1j) * (t[m - 1] - t[m]) / 2)
    tmp1 = np.eye(N - 1) - tmp * (func_y_psi(u, q))
    w_1 = np.dot(inv(tmp1), func_psi(psi[m, :],u, (t[m] + t[m - 1])/2,q)).real
    tmp2 = (t[m - 1] - t[m]) * w_1
    psi[m - 1] = psi[m] + np.transpose(tmp2)
    y[m - 1, 0] = u_left
    y[m - 1, 1:N] = psi[m - 1]
    y[m - 1, N] = u_right


fig2 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(a, b), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=1, color='red')
line2, = ax.plot([], [], lw=1, color='green')


def animate(i):
    line.set_xdata(x)
    line.set_ydata(y[i, :])
    line2.set_xdata(x)
    line2.set_ydata(u_model(x,t[i]))
    return line,line2

anim = animation.FuncAnimation(fig2, animate, frames=N+M, interval=50)
plt.show()
