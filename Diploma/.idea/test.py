import numpy as np
N = 20
a, b = 0, 1
i = np.linspace(a, b, N + 1)

list1 = []
list1 = list(map(lambda i : 2 * i - 1 + 2 * np.sin(5 * i * np.pi) + 0.35, x))


init_q = []
def q_init(x):
    return 2 * x - 1 + 2 * np.sin(5 * x * np.pi) + 0.35

for n in range(0, N + 1):
    init_q.append(q_init(i[n]))
print(init_q)