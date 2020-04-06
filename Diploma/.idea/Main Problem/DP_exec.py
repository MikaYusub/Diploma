import numpy as np
from DiplomaDirectProblem import DirectProblem

eps = 0.05;
M = 200;
N = 200
u_left = -8;
u_right = 4
a = 0;
b = 1.
T = 0.2;
t_0 = 0.
tau = (T - t_0) / M
h = (b - a) / N
t = np.linspace(t_0, T, M + 1)
x = np.linspace(a, b, N + 1)

init_q = list(map(lambda i : 2 * i - 1 + 2 * np.sin(5 * i * np.pi) + 0.35, x))

u = DirectProblem.direct_problem(a, b, eps, M, N, u_left, u_right, t, x, init_q, h,tau)

