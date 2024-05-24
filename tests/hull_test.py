import numpy as np
from scipy.optimize import linprog

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

n_points = 10000
n_dim = 10
Z = np.random.rand(n_points,n_dim)
x = np.random.rand(n_dim)
print(Z, x)
print(in_hull(Z, x))