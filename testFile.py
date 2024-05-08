import numpy as np

def func(a, b):
    return np.array([1, 1, 0, 0, 5])

# Generate list of lambda functions
lambda_functions = [lambda arr, idx=i: arr[idx] for i in range(5)]

# Test the lambda functions
results = [f(func(1, 2)) for f in lambda_functions]

lambda_functions[0]()
print(results)
