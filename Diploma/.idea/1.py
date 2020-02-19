import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags

def thomas(a, b, c, d):

    # Finding the size of the matrix and determine n
    n = len(b)
    # print n # Used for debugging
    # Test the size of a and c
    if len(a) != n - 1:
        print
        'Wrong index size for a.\n A should have an index of', n - 1, '\n Your a has ', len(a)
        exit()
    if len(c) != n - 1:
        print
        'Wrong index size for c.\n C should have an index of', n - 1, '\n Your c has', len(c)
        exit()

    # Converting to float and appending 0.0 to c
    for i in range(0, len(a)):
        a[i] = float(a[i])
    for i in range(0, len(b)):
        b[i] = float(b[i])
    for i in range(0, len(c)):
        c[i] = float(c[i])
    for i in range(0, len(d)):
        d[i] = float(d[i])

    # c.append(0.0)  # Hack to make the function to work
    np.append(c,0.0)
    # Calculate p and q
    p = []
    q = []
    np.append(p,c[0] / b[0])
    np.append(q,d[0] / b[0])

    for j in range(1, n):
        pj = c[j] / (b[j] - a[j - 1] * p[j - 1])
        qj = (d[j] - a[j - 1] * q[j - 1]) / (b[j] - a[j - 1] * p[j - 1])
        # p.append(pj)
        np.append(p,pj)
        # q.append(qj)
        np.append(q, qj)

    # print p,q # Used for debugging the code!

    # Back sub
    x = [];
    # x.append(q[n - 1])
    np.append(x, q[n-1])

    for j in range(n - 2, -1, -1):
        xj = q[j] - p[j] * x[0]  # Value holder
        np.insert(x,0,xj)
        # x.insert(0, xj)  # Building the list backwards

    # Return the value
    return x

def MyTDMAsolver(aa, bb, cc, B):

    n = len(B)
    Ab = np.zeros((3, n))
    Ab[0, 1:] = cc[:-1]
    Ab[1, :] = aa
    Ab[2, :-1] = bb[1:]
    X = solve_banded((1,1),Ab,B)
    return X


def tdma(a, b, c, d):

    n = len(b)
    x = np.zeros(n)

    # elimination:

    for k in range(1, n):
        q = a[k] / b[k - 1]
        b[k] = b[k] - c[k - 1] * q
        d[k] = d[k] - d[k - 1] * q

    # backsubstitution:

    q = d[n - 1] / b[n - 1]
    x[n - 1] = q

    for k in range(n - 2, -1, -1):
        q = (d[k] - c[k] * q) / b[k]
        x[k] = q

    return x

def TDMAsolver(a, b, c, d):
    
    nf = len(a)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy the array
    for it in range(1, nf):
        mc = ac[it] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = ac
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    del bc, cc, dc  # delete variables from memory

    return xc


def tripiv(a, b, c, d):
    """Solution of a linear system of algebraic equations with a
        tri-diagonal matrix of coefficients using the Thomas-algorithm with pivoting.

    Args:
        a(array): an array containing lower diagonal (a[0] is not used)
        b(array): an array containing main diagonal
        c(array): an array containing lower diagonal (c[-1] is not used)
        d(array): right hand side of the system
    Returns:
        x(array): solution array of the system

    """

    n = len(b)
    x = np.zeros(n)
    fail = 0

    # reordering

    a[0] = b[0]
    b[0] = c[0]
    c[0] = 0

    # elimination:

    l = 0

    for k in range(0, n):
        q = a[k]
        i = k

        if l < n - 1:
            l = l + 1

        for j in range(k + 1, l + 1):
            q1 = a[j]
            if (np.abs(q1) > np.abs(q)):
                q = q1
                i = j
        if q == 0:
            fail = -1

        if i != k:
            q = d[k]
            d[k] = d[i]
            d[i] = q
            q = a[k]
            a[k] = a[i]
            a[i] = q
            q = b[k]
            b[k] = b[i]
            b[i] = q
            q = c[k]
            c[k] = c[i]
            c[i] = q
        for i in range(k + 1, l + 1):
            q = a[i] / a[k]
            d[i] = d[i] - q * d[k]
            a[i] = b[i] - q * b[k]
            b[i] = c[i] - q * c[k]
            c[i] = 0

    # backsubstitution

    x[n - 1] = d[n - 1] / a[n - 1]
    x[n - 2] = (d[n - 2] - b[n - 2] * x[n - 1]) / a[n - 2]

    for i in range(n - 3, -1, -1):
        q = d[i] - b[i] * x[i + 1]
        x[i] = (q - c[i] * x[i + 2]) / a[i]

    return x

def TDMA(a, b, c, f):
    # a, b, c, f = map(lambda k_list: map(float, k_list), (a, b, c, f))
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0] * n

    for i in range(n - 1):
        alpha.append(-b[i] / (a[i] * alpha[i] + c[i]))
        beta.append((f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i]))

    x[n - 1] = (f[n - 1] - a[n - 2] * beta[n - 1]) / (c[n - 1] + a[n - 2] * alpha[n - 1])

    for i in reversed(range(n - 1)):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x

a = np.array([0,3.,1,3])
b = np.array([10.,10.,7.,4.])
c = np.array([2.,4.,5.,0])
d = np.array([3,4,5,6.])

A = np.array([[10,2,0,0],
              [3,10,4,0],
              [0,1,7,5],
              [0,0,3,4]],dtype=float)

# nn = len(d)
# diagonals = [a, b, c]
# banded_matrix = diags(diagonals, [0, -1, 1], shape=(nn, nn)).toarray()
# a = [-1.5997515 -1.5997515j  -1.59950533-1.59950533j -1.5992617 -1.5992617j
#  -1.59902061-1.59902061j -1.59878207-1.59878207j -1.59854611-1.59854611j
#  -1.59831275-1.59831275j -1.59808202-1.59808202j -1.59785398-1.59785398j
#  -1.59762869-1.59762869j -1.59740623-1.59740623j -1.59718672-1.59718672j]

print(MyTDMAsolver(a,b,c,d))

