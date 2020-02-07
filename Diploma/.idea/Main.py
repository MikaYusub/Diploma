import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve
import matplotlib.animation as animation
import time

start_time = time.time()
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'

eps = 0.3
M = 50
N = 50
u_left = -8
u_right = 4
a = 0
b = 1.
T = 0.2
t_0 = 0.
h = (b - a) / N
tau = (T - t_0) / M
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)
init_q = []
S = 100  # Количество итераций
ss = np.linspace(0,S-2,S-1)
q = np.zeros((S, N + 1))
J = np.zeros(S)
beta = 25

def q_init(x):
     #2 * x - 1 + 2 * np.sin(5 * x * np.pi) + 0.35
     return np.sin(3*x*np.pi)


for n in range(0, N + 1):
    init_q.append(q_init(x[n]))


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
                  )
        for n in range(1, N - 2):
            f.itemset(n,
                      (eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2)
                      + (y[n] * (y[n + 1] - y[n - 1]) / (2 * h))
                      - y[n] * q[n + 1]
                      )
        f.itemset(N - 2,
                  (eps * (u_right - 2 * y[N - 2] + y[N - 3]) / h ** 2)
                  + (y[N - 2] * (u_right - y[N - 3]) / (2 * h))
                  - y[N - 2] * q[N - 1]
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
        w_1 = solve(tmp1, func(y[m, :], (t[m] + t[m + 1]) / 2, x, q))
        tmp2 = (t[m + 1] - t[m]) * w_1.real
        y[m + 1] = y[m] + np.transpose(tmp2)
        u[m + 1, 1:N] = y[m + 1]
        u[m + 1, 0] = u_left
        u[m + 1, N] = u_right
    return u


def conjucate_problem(eps, M, N, a, b, T, t_0, t, x, q, h, u, f_obs):
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
                      + (u[n+1] * (y[n + 1] - y[n - 1]) / (2 * h))
                      + y[n] * q[n+1]
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
            f_y.itemset((n, n - 1), -eps / h ** 2 - u[n+1] / (2 * h))
        for n in range(0, N - 1):
            f_y.itemset((n, n), 2 * eps / h ** 2 + q[n+1])
        for n in range(0, N - 2):
            f_y.itemset((n, n + 1), (-eps / h ** 2) + (u[n+1] / (2 * h)))
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


def gradient_calculation(u, psi, tau, M, N):
    res = np.zeros(N+1)
    for n in range(N+1):
        for m in range(1, M + 1):
            res[n] = (u[m, n] * psi[m, n] + u[m - 1, n] * psi[m - 1, n]) * tau / 2
    return res


def functional_calculation(u, f_obs, h, N):
    res = 0
    for n in range(1, N + 1):
        res += ((u[n] - f_obs[n]) ** 2 + (u[n - 1] - f_obs[n - 1]) ** 2) * h / 2
    return res


tmp = direct_problem(eps, M, N, a, b, u_left, u_right, T, t_0, t, x, init_q, h)

f_obs = tmp[M, :]
# q[0,:]=
for s in range(S-1):  ## while -> condition
    print(s)
    u = direct_problem(eps, M, N, a, b, u_left, u_right, T, t_0, t, x, q[s, :], h)
    J[s] = functional_calculation(u[M, :], f_obs, h, N)
    psi = conjucate_problem(eps, M, N, a, b, T, t_0, t, x, q[s, :], h, u, f_obs)
    dJ = gradient_calculation(u, psi, tau, M, N)
    q[s + 1, :] = q[s, :] - beta * dJ


# ax = plt.axes(xlim=(a, b), ylim=(u_left, u_right))
# def animate(i):
#     line.set_xdata(x)
#     line.set_ydata(u[i,:])
#     line2.set_xdata(x)
#     line2.set_ydata(u_init(x))
#     return line,line2
#
# anim = animation.FuncAnimation(fig2, animate, frames=1+M, interval=50)
# # FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
# # anim.save(r'C:\Users\FS\Desktop\Main Mission\Conjucate_problem_solution.mp4', writer=FFwriter)
# plt.show()
fig2 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
line, = ax.plot([], [], lw=1, color='red')
line2, = ax.plot([], [], lw=1, color='green')

def animate(i):
    line.set_xdata(x)
    line.set_ydata(q[i,:])
    line2.set_xdata(x)
    line2.set_ydata(q_init(x))
    return line

anim = animation.FuncAnimation(fig2, animate)
# FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
# anim.save(r'C:\Users\FS\Desktop\Main Mission\Conjucate_problem_solution.mp4', writer=FFwriter)
plt.show()

plt.plot(J[0:S-1])
plt.xlim((0,S-2))
plt.ylim((0,0.35))
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
