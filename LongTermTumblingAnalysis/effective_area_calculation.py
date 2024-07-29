import numpy as np
from numpy import cos, sin, arctan2, sqrt
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
from numba import jit
from scipy.spatial.transform import Rotation as R
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)


def set_of_coefficients(p, n0, omega_x, omega_y):
    nx0, ny0, nz0 = n0
    A, B, C, D = p

    k = np.sqrt(omega_x**2 + omega_y**2)
    eq1 = nz0 - A
    eq2 = nx0 - (-B * omega_y / k + C)
    eq3 = ny0 - (B * omega_x / k + D)
    eq4 = -omega_y * C + omega_x * D
    return (eq1, eq2, eq3, eq4)

@jit(nopython=True, cache=True)
def n(t, n0, omega_v):
    x0, y0, z0 = n0
    om1, om2, om3 = omega_v
    k = np.linalg.norm(omega_v)

    #k = np.sqrt(omega_x ** 2 + omega_y ** 2)
    #nx = (omega_y / k) * (A * np.sin(k * t) - B * np.cos(k*t)) + C
    #ny = (-omega_x / k) * (A * np.sin(k * t) - B * np.cos(k * t)) + D
    #nz = A * np.cos(k * t) + B * np.sin(k * t)

    #n = np.array([nx, ny, nz])
    n = np.zeros((3,))
    n[0] = (2 * om1 ** 2 * x0 - om1 * om2 * y0 * cos(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 7 * arctan2(0,
                                                                                                          om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om1 * om2 * y0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 11 * arctan2(0,
                                                              om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om1 * om2 * y0 - 2 * om1 * om3 * z0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2)) * cos(
        5 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om1 * om3 * z0 + om2 ** 2 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 7 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om2 ** 2 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 11 * arctan2(0,
                                                              om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om2 * z0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2)) * cos(
        5 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om3 ** 2 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 7 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om3 ** 2 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 11 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om3 * y0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 7 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om3 * y0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 11 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2)) / (
                       2 * (om1 ** 2 + om2 ** 2 + om3 ** 2))
    n[1] = (om1 ** 2 * y0 * cos(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 5 * arctan2(0,
                                                                                     om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om1 ** 2 * y0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 9 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om1 * om2 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 5 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om1 * om2 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 9 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om1 * om2 * x0 - 2 * om1 * z0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2)) * cos(
        3 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om2 ** 2 * y0 - 2 * om2 * om3 * z0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2)) * cos(
        3 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om2 * om3 * z0 + om3 ** 2 * y0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 5 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om3 ** 2 * y0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 9 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om3 * x0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 5 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om3 * x0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 9 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2)) / (
                       2 * (om1 ** 2 + om2 ** 2 + om3 ** 2))
    n[2] = (2 * om1 ** 2 * z0 * cos(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2)) - om1 * om3 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 3 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om1 * om3 * x0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 5 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om1 * om3 * x0 + om1 * y0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 3 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + om1 * y0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 5 * arctan2(0,
                                                                                                   om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om2 ** 2 * z0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2)) - om2 * om3 * y0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 3 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om2 * om3 * y0 * cos(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 5 * arctan2(0,
                                                             om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om2 * om3 * y0 - om2 * x0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(
        t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) - 3 * arctan2(0, om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) - om2 * x0 * sqrt(
        om1 ** 2 + om2 ** 2 + om3 ** 2) * sin(t * sqrt(om1 ** 2 + om2 ** 2 + om3 ** 2) + 5 * arctan2(0,
                                                                                                   om1 ** 2 + om2 ** 2 + om3 ** 2) / 2) + 2 * om3 ** 2 * z0) / (
                       2 * om1 ** 2 + 2 * om2 ** 2 + 2 * om3 ** 2)
    #print(n)
    return n/np.linalg.norm(n)

def n_s(t, omega_sun=2 * np.pi / (3600 * 24 * 365.2425)):
    return np.array([np.sin(omega_sun * t), 0, -np.cos(omega_sun*t)])


def SRP_effect(t, n0, omega_vec):
    current_ns = n_s(t)
    current_n = n(t, n0, omega_vec)

    cos_theta = np.dot(current_n, -current_ns)
    #print(current_ns)
    val = abs(cos_theta) * cos_theta * current_n
    return val


new_z = np.array([0, 0, 1])/np.linalg.norm(np.array([0, 0, 1]))     # aka Z_B


#generic_perp_vec = np.array([4, 2, 5])/np.linalg.norm(np.array([4, 2, 5]))
#new_y = np.cross(generic_perp_vec, new_z)/np.linalg.norm(np.cross(generic_perp_vec, new_z))

new_y = np.array([0, 1, 0])
new_x = np.cross(new_y, new_z)/np.linalg.norm(np.cross(new_y, new_z))
initial_body_to_inertial = np.zeros((3, 3))
initial_body_to_inertial[:, 0] = new_x
initial_body_to_inertial[:, 1] = new_y
initial_body_to_inertial[:, 2] = new_z
initial_body_to_inertial = np.dot(R.from_euler('z', 45, degrees=True).as_matrix(), initial_body_to_inertial)
print(initial_body_to_inertial)

n_initial = initial_body_to_inertial[:, 2]


SRPx = lambda t:  SRP_effect(t, n_initial, omega_inertial)[0]
SRPy = lambda t:  SRP_effect(t, n_initial, omega_inertial)[1]
SRPz = lambda t:  SRP_effect(t, n_initial, omega_inertial)[2]

om_v_list = [(5, 5, 0),
             (5, -5, 0)]

color_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]

for p_id in range(2):
    plt.figure()
    for id, om_v in enumerate(om_v_list):
        # but here omega_x and omega_y are expressed in the body-fixed frame
        omega_x = np.deg2rad(om_v[0])
        omega_y = np.deg2rad(om_v[1])
        omega_z = np.deg2rad(om_v[2])

        omega_inertial = np.dot(initial_body_to_inertial, np.array([omega_x, omega_y, omega_z]))

        print(omega_inertial)
        #A, B, C, D = fsolve(set_of_coefficients, np.array([1, 1, 1, 1]), (n_initial, omega_x, omega_y))

        #surface_normal_coeffs = (A, B, C, D)
        #print(surface_normal_coeffs)

        attitude_motion_period = 10*2 * np.pi / (np.sqrt(omega_x**2 + omega_y**2 + omega_z**2))

        t_arr = np.linspace(0, 365*24*3600, 500000)
        SRPx_list = []
        SRPy_list = []
        SRPz_list = []
        for t in t_arr:
            SRPx_list.append(SRPx(t))
            SRPy_list.append(SRPy(t))
            SRPz_list.append(SRPz(t))
        #plt.figure()
        #lt.plot(t_arr, SRPx_list)

        #plt.figure()
        #plt.plot(t_arr, SRPy_list)
        if (p_id==0):
            plt.plot(t_arr/(24*3600), SRPz_list, label=r'$\vec{\omega}_{\textit{B}}$ =' + f'{list(om_v)}', color=color_list[id])
        else:
            plt.plot(t_arr / (24 * 3600), SRPx_list, label=r'$\vec{\omega}_{\textit{B}}$ =' + f'{list(om_v)}', color=color_list[id])
    plt.grid(True)
    plt.xlabel(r'$t$ [days]', fontsize=14)
    if (p_id == 0):
        plt.ylabel(r'$\tilde{f}_{SRP, z}$  [-]', fontsize=14)
    else:
        plt.ylabel(r'$\tilde{f}_{SRP, x}$  [-]', fontsize=14)
    plt.legend()
    #plt.savefig(f'/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc/simplified_attitude_{p_id}.png',
    #                dpi=600,
    #                bbox_inches='tight')




"""
quad_results = quad(SRPx, 0, attitude_motion_period, limit=100000)
effective_area = quad_results[0]/attitude_motion_period
effective_error = quad_results[1]/attitude_motion_period
print(quad_results)
print(effective_area)
print(effective_error)

quad_results = quad(SRPy, 0,attitude_motion_period, limit=100000)
effective_area = quad_results[0]/attitude_motion_period
effective_error = quad_results[1]/attitude_motion_period
print(quad_results)
print(effective_area)
print(effective_error)

quad_results = quad(SRPz, 0, attitude_motion_period, limit=100000)
effective_area = quad_results[0]/attitude_motion_period
effective_error = quad_results[1]/attitude_motion_period
print(quad_results)
print(effective_area)
print(effective_error)
"""


plt.show()