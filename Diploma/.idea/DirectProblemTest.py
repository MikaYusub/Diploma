import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.animation as animation

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
