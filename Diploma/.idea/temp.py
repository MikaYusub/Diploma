import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve
import matplotlib.animation as animation

eps = 0.03
M = 200
N = 200
u_left = -8
u_right = 4
a = 0
b = 1.
T = 0.2
t_0 = 0.
h = (b - a) / N
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)
q = [];


def tmp_f(x, t):
    tmp_var = -eps * (1 - 2 * t) * np.sin(x) \
              + 2 * np.sin(x) \
              + ((1 - 2 * t) ** 2) * np.sin(x) * np.cos(x) \
              - np.sin(3 * x) * (1 - 2 * t) * np.sin(x)
    return tmp_var


def q_init(x):
    return 2 * x - 1 + 2 * np.sin(5 * x * np.pi) + 0.35


for n in range(0, N + 1):
    q.append(q_init(x[n]))


def u_init(x):
    return (x ** 2 - x - 2) \
           - 6 * np.tanh((-3 * x + 0.75) / eps)


def direct_problem(eps, M, N, a, b, u_left, u_right, T, t_0, t, x, q, h):
    def func(y, t, x, q):
        f = np.zeros((N - 1, 1))
        f.itemset(0,
                  (eps * (y[1] - 2 * y[0] + u_left) / h ** 2)
                  + (y[0] * (y[1] - u_left) / (2 * h))
                  - y[0] * q[1]
                  # - tmp_f(x[1], t)
                  )
        for n in range(1, N - 2):
            f.itemset(n,
                      (eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2)
                      + (y[n] * (y[n + 1] - y[n - 1]) / (2 * h))
                      - y[n] * q[n + 1]
                      # - tmp_f(x[n + 1], t)
                      )
        f.itemset(N - 2,
                  (eps * (u_right - 2 * y[N - 2] + y[N - 3]) / h ** 2)
                  + (y[N - 2] * (u_right - y[N - 3]) / (2 * h))
                  - y[N - 2] * q[N - 1]
                  # - tmp_f(x[N - 1], t)
                  )
        return f

    def func_y(y, q):
        f_y = np.zeros((N - 1, N - 1), dtype='float64')
        f_y.itemset((0, 0),
                    (-2 * eps / h ** 2)
                    + ((y[1] - u_left) / (2 * h))
                    - q[1])
        for n in range(1, N - 1):
            f_y.itemset((n, n - 1),
                        eps / h ** 2 - y[n] / (2 * h))
        for n in range(1, N - 2):
            f_y.itemset((n, n),
                        -2 * eps / h ** 2
                        + ((y[n + 1] - y[n - 1]) / (2 * h)) - q[n + 1])
        for n in range(0, N - 2):
            f_y.itemset((n, n + 1),
                        eps / h ** 2 + y[n] / (2 * h))

        f_y.itemset((-1, -1),
                    -2 * eps / h ** 2
                    + ((u_right - y[N - 3]) / (2 * h))
                    - q[N - 1])
        return f_y

    u = np.zeros((M + 1, N + 1))
    y = np.zeros((M + 1, N - 1))
    for n in range(N + 1):
        u[0, n] = u_init(x[n])
    y[0, :] = u[0, 1:N]

    for m in range(M):
        tmp = ((1 + 1j) * (t[m + 1] - t[m]) / 2)
        tmp1 = np.eye(N - 1) - tmp * (func_y(y[m, :], q))
        w_1 = np.dot(inv(tmp1), func(y[m, :], (t[m] + t[m + 1]) / 2, x, q)).real
        tmp2 = (t[m + 1] - t[m]) * w_1
        y[m + 1] = y[m] + np.transpose(tmp2)
        u[m + 1, 1:N] = y[m + 1]
        u[m + 1, 0] = u_left
        u[m + 1, N] = u_right

    return u


def conjucate_problem(eps, M, N, t, x, q, h, u, f_obs):
    def func_psi(y, u, t, q):
        f = np.zeros((N - 1, 1))
        f.itemset(0,
                  (-eps * (y[1] - 2 * y[0]) / h ** 2)
                  + (u[1] * y[1] / (2 * h))
                  + y[0] * q[1]
                  )
        for n in range(1, N - 2):
            f.itemset(n,
                      (-eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2)
                      + (u[n + 1] * (y[n + 1] - y[n - 1]) / (2 * h))
                      + y[n] * q[n + 1]
                      )
        f.itemset(N - 2,
                  (-eps * (-2 * y[N - 2] + y[N - 3]) / h ** 2)
                  - (u[N - 1] * y[N - 3] / (2 * h))
                  + y[N - 2] * q[N - 1]
                  )
        return f

    def func_y_psi(u, q):
        f_y = np.zeros((N - 1, N - 1), dtype='float64')
        for n in range(1, N - 1):
            f_y.itemset((n, n - 1), -eps / h ** 2 - u[n + 1] / (2 * h))
        for n in range(0, N - 1):
            f_y.itemset((n, n), 2 * eps / h ** 2 + q[n + 1])
        for n in range(0, N - 2):
            f_y.itemset((n, n + 1), (-eps / h ** 2) + (u[n + 1] / (2 * h)))
        return f_y

    psi = np.zeros((M + 1, N + 1))
    y = np.zeros((M + 1, N - 1))

    psi[M, :] = -2 * (u[M, :] - f_obs)
    y[M, :] = psi[M, 1:N]
    for m in range(M, 0, -1):
        tmp = ((1 + 1j) * (t[m - 1] - t[m]) / 2)
        tmp1 = np.eye(N - 1) - tmp * (func_y_psi(u[m, :], q))
        w_1 = solve(tmp1, func_psi(y[m, :], u[m, :], (t[m] + t[m - 1]) / 2, q))
        tmp2 = (t[m - 1] - t[m]) * w_1.real
        y[m - 1, :] = y[m, :] + np.transpose(tmp2)
        psi[m - 1, 1:N] = y[m - 1, :]
        psi[:, 0] = 0
        psi[:, N] = 0

    return psi


tmp = direct_problem(eps, M, N, a, b, u_left, u_right, T, t_0, t, x, q, h)

f_obs = tmp[M, :]

psi = conjucate_problem(eps, M, N, t, x, q, h, tmp, f_obs)

fig2 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(a, b), ylim=(-9, 5))
line, = ax.plot([], [], lw=1, color='red')
line2, = ax.plot([], [], lw=1, color='green')


def animate(i):
    line.set_xdata(x)
    line.set_ydata(psi[i, :])
    line2.set_xdata(x)
    line2.set_ydata(u_init(x))
    return line, line2


anim = animation.FuncAnimation(fig2, animate, frames=M + 1, interval=50)
plt.show()
