import numpy as np
N = 50
f_y = np.zeros((N - 1, 1))

f_y.itemset((0, 0), 100)
for n in range(1, N - 2):
    f_y.itemset(n, 10)
f_y.itemset((N - 2), 22)

#print(f_y)

x = []
h = 1/N
a=0
for n in range(N + 1):
    x.append(a + h * n)
    print(x[n])
print(x[0])
