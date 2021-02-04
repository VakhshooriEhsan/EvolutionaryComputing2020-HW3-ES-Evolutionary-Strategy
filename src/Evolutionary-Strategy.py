import numpy as np

def Reliability(r, n):
    return 1 - (1-r)**n

# Initial values:
m = 5
w = np.array([7.0, 8.0, 8.0, 6.0, 9.0])
wv2 = np.array([1.0, 2.0, 3.0, 4.0, 2.0])
V = 110
C = 175
W = 200
alpha = np.array([2.33, 1.45, 0.541, 8.05, 1.95]) / 10**5
beta = np.array([1.5, 1.5, 1.5, 1.5, 1.5])

# Conditions:
def g1(r, n):
    return np.sum(wv2 * n**2) - V <= 0

def g2(r, n):
    return np.sum(alpha * (-1000/np.log(r))**beta * (n + np.exp(0.25*n))) - C <= 0

def g3(r, n):
    return np.sum(w*n*np.exp(0.25*n)) - W <= 0

# Fitness:
def f1(r, n):
    R = Reliability(r, n)
    return R[1]*R[2] + R[3]*R[4] + R[1]*R[4]*R[5] + R[2]*R[3]*R[5] - R[1]*R[2]*R[3]*R[4] - R[1]*R[2]*R[3]*R[5] - R[1]*R[2]*R[4]*R[5] - R[1]*R[3]*R[4]*R[5] - R[2]*R[3]*R[4]*R[5] + 2*R[1]*R[2]*R[3]*R[4]*R[5]

def f2(r, n):
    R = Reliability(r, n)
    return np.prod(R)

# ------------------------------ Complex (bridge) system (P1) ------------------------------

n = np.zeros(m)
r = np.zeros(m)