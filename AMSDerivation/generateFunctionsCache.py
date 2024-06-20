from fullEllipseCoefficientsFunctions import ellipse_full_coefficients_function_shadow_FALSE_ideal_model, ellipse_full_coefficients_function_shadow_TRUE_ideal_model
from truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_ideal_model, ellipse_truncated_coefficients_function_shadow_TRUE_ideal_model
from time import time

funcs = ellipse_full_coefficients_function_shadow_FALSE_ideal_model([-1, -1, -1, -1, -1, -1])
t0 = time()
l = funcs[0](0, 1)
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(l)
print(time()-t0)


funcs = ellipse_truncated_coefficients_function_shadow_FALSE_ideal_model()
t0 = time()
l = funcs[0](0, 1)
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(l)
print(time()-t0)


funcs = ellipse_full_coefficients_function_shadow_TRUE_ideal_model([-1, -1, -1, -1, -1, -1])
t0 = time()
l = funcs[0](0, 1)
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(l)
print(time()-t0)


funcs = ellipse_truncated_coefficients_function_shadow_TRUE_ideal_model()
t0 = time()
l = funcs[0](0, 1)
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(l)
print(time()-t0)
