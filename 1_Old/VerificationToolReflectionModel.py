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
#print(v)
#print(v/np.linalg.norm(v))


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

#print(v)
#print(v/np.linalg.norm(v))

ns = np.longdouble(np.array([-1, -1, -1]))
ns = ns/np.linalg.norm(ns)
n = np.longdouble(np.array([0, 0, 1]))
n = n/np.linalg.norm(n)

alpha_front = 0.5
rho_s_front = 0.2
rho_d_front = 0.3
emissivity_front = 0.9
B_front = 0.3

alpha_back = 0.1
rho_s_back = 0.8
rho_d_back = 0.1
emissivity_back = 0.1
B_back = 0.8

absorption_reemission_ratio = (emissivity_back * B_back - emissivity_front * B_front) / (emissivity_back + emissivity_front)

c_theta = np.dot(n, -ns) / (
        np.linalg.norm(n) * np.linalg.norm(ns))
print(c_theta)
# Get the vane torque according to the optical model, in the body frame
if (c_theta >= 0):  # the front is exposed
    # W * vane_area/ c_sol *
    f = (abs(c_theta)) * ((alpha_front * absorption_reemission_ratio - 2 * rho_s_front * c_theta - rho_d_front * B_front) * n + (
                                  alpha_front + rho_d_front) * ns)  # was a minus for the self.sun_direction_body_frame
else:
    # W * vane_area/ c_sol *
    f = (abs(c_theta)) * ((alpha_back * absorption_reemission_ratio - 2 * rho_s_back * c_theta + rho_d_back * B_back) * n + (
                                  alpha_back + rho_d_back) * ns)


print(f)
#print((0.1 * (ns + n * (0.1*0.8 - 0.3*0.9)/(0.1+0.9)) - 2 * 0.8 * -1/np.sqrt(3) * n + 0.1 * (ns + 0.8*n)) * abs(-1/np.sqrt(3)))
print((0.5 * (ns + n * (0.1*0.8 - 0.3*0.9)/(0.1+0.9)) - 2 * 0.2 * 1/np.sqrt(3) * n + 0.3 * (ns - 0.3*n)) * abs(1/np.sqrt(3)))