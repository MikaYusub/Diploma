import numpy as np
from DiplomaDirectProblem import DirectProblem
from DiplomaConjugateProblem import ConjugateProblem
from utilities import Utils
import matplotlib.pyplot as plt

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
init_q = []
S = 401  # Количество итераций
q = np.zeros((S, N + 1))
beta = 0.125
J = np.zeros(S)

init_q = list(map(lambda i: 2 * i - 1 + 2 * np.sin(5 * i * np.pi) + 0.35, x))
u_init = list(map(lambda i: (i ** 2 - i - 2) - 6 * np.tanh((-3 * i + 0.75) / eps), x))

u = DirectProblem.direct_problem(a, b, eps, M, N, u_left, u_right, t, x, init_q, h, tau, u_init)
f_obs = u[M, :]
for s in range(S - 1):  ## while -> condition
    print(s)
    u = DirectProblem.direct_problem(a, b, eps, M, N, u_left, u_right, t, x, q[s, :], h, tau, u_init)
    J[s] = Utils.functional_calculation(u[M, :], f_obs, h, N)
    psi = ConjugateProblem.conjucate_problem(eps, M, N, t, x, q[s, :], h, u, f_obs, tau)
    dJ = Utils.gradient_calculation(u, psi, tau, M, N)
    q[s + 1, :] = q[s, :] - beta * dJ
Utils.DrawConjugate(psi, x, M)
