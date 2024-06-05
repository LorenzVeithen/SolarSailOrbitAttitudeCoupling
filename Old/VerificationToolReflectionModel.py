import numpy as np

ns = np.longdouble(np.array([-1, -1, -1]))
ns = ns/np.linalg.norm(ns)
n = np.longdouble(np.array([0, 0, 1]))
n = n/np.linalg.norm(n)

a = 0.2
b = 0.2 * (0.92*0.79 - 0.55 * 0.3)/(0.92 + 0.3)
c = -2 * 0.7 * (np.dot(-ns, n)/(np.linalg.norm(ns) * np.linalg.norm(n)))
d = 0.1
e = - 0.1 * 0.55
v = a * ns + b * n + c * n + d * ns + e * n
np.set_printoptions(precision=20)
print(v)
print(v/np.linalg.norm(v))


ns = np.longdouble(np.array([1, 1, 1]))
ns = ns/np.linalg.norm(ns)
n = np.longdouble(np.array([0, 0, 1]))
n = n/np.linalg.norm(n)

a = 0.5
b = 0.5 * (0.92*0.79 - 0.55 * 0.3)/(0.92 + 0.3)
c = -2 * 0.2 * (np.dot(-ns, n)/(np.linalg.norm(ns) * np.linalg.norm(n)))
d = 0.3
e = 0.3 * 0.79
v = a * ns + b * n + c * n + d * ns + e * n

v = a * ns + b * n + c * n + d * ns + e * n

print(v)
print(v/np.linalg.norm(v))

