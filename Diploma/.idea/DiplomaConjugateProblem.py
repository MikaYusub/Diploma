import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1,2,3])
c = np.matmul(a, b)
d = np.dot(a,b)
print(c, d)
