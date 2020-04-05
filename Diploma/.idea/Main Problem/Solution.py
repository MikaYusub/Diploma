from DiplomaDirectProblem import DirectProblem
from DiplomaConjugateProblem import ConjugateProblem
from utilities import Utils
import numpy as np

eps = 0.03
M = 200
N = 300
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
S = 33# Количество итераций
q = np.zeros((S, N + 1))
J = np.zeros(S)
beta = 0.125
f_obs = []

init_q = list(map(lambda i : 2 * i - 1 + 2 * np.sin(5 * i * np.pi) + 0.35, x))
tmp = DirectProblem.direct_problem(eps, M, N, u_left, u_right, t, x, init_q, h, tau)
f_obs = tmp[M, :]

for s in range(S - 1):  ## while -> condition
    print(s)
    u = DirectProblem.direct_problem(eps, M, N, u_left, u_right, t, x, q[s, :], h, tau)
    J[s] = Utils.functional_calculation(u[M, :], f_obs, h, N)
    psi = ConjugateProblem.conjucate_problem(eps, M, N, t, x, q[s, :], h, u, f_obs,tau)
    dJ = Utils.gradient_calculation(u, psi, tau, M, N)
    q[s + 1, :] = q[s, :] - beta * dJ

Utils.DrawSolution(S,x,q,J,init_q)
