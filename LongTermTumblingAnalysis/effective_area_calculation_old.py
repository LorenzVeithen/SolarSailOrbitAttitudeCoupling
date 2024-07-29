import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def single_axis_period(Tx):
    return 4/(3*np.pi)

def sa(t):
    return abs(np.cos(10 * t)) * np.cos(10 * t)**2

def two_axis_effective(Tx, Ty):
    def func(t):
        return abs((np.cos(2 * np.pi * t/Tx) * np.cos(2 * np.pi * t/Ty))**3)

    period = np.lcm(Tx, Ty)
    # integrate numerically
    integration_output = integrate.quad(func, 0, period, limit=1000)
    print(integration_output)
    return integration_output[0]/period

print(integrate.quad(sa, 0, 2 * np.pi/10)[0]/(2 * np.pi/10))