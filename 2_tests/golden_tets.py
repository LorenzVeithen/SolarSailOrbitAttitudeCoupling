from scipy.optimize import golden
import numpy as np
def f(t):
    return (t+0.5)**2

print(golden(f, brack=(-1, 1), tol=1e-7))