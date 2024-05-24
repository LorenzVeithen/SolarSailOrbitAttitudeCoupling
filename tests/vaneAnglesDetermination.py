# Considering a single vane for a given torque, let's determine the right angles of the vanes
from constants import *
import numpy as np
from scipy.optimize import fsolve, minimize, direct, Bounds
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
from time import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from MiscFunctions import *
from ACS_dynamical_models import vane_dynamical_model

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

vane_inner_angle = (np.pi - (2 * vane_angle))/2
h_vane = np.cos(vane_inner_angle) * vane_side_length
b_vane = 2 * np.sin(vane_inner_angle) * vane_side_length
A_vane = h_vane * b_vane/2
c = 299792458   # m/s

alpha_front = vanes_optical_properties[0][0]
alpha_back = vanes_optical_properties[0][1]
rho_s_front = vanes_optical_properties[0][2]
rho_s_back = vanes_optical_properties[0][3]
rho_d_front = vanes_optical_properties[0][4]
rho_d_back = vanes_optical_properties[0][5]
B_front = vanes_optical_properties[0][6]
B_back = vanes_optical_properties[0][7]
emissivity_front = vanes_optical_properties[0][8]
emissivity_back = vanes_optical_properties[0][9]

vanes_origin = vanes_origin_list[0]
vanes_coordinates = vanes_coordinates_list[0]
R_BV = vanes_rotation_matrices_list[0]
R_VB = np.linalg.inv(R_BV)
absorption_reemission_ratio = (emissivity_back * B_back - emissivity_front * B_front)/(emissivity_back + emissivity_front)

sun_angle_alpha = np.deg2rad(75)
sun_angle_beta = np.deg2rad(24)
n_s = -np.array([np.sin(sun_angle_alpha) * np.cos(sun_angle_beta), np.sin(sun_angle_alpha) * np.sin(sun_angle_beta), -np.cos(sun_angle_alpha)])   # In the body reference frame
n_s = np.array([1, 1, 1])
n_s = n_s/np.linalg.norm(n_s)
W = 1400    # W / m^2 - roughly
initial_centroid = np.array([(2 * h_vane/3), 0, 0])

vstack_stacking = np.array([[0, 0, 0]])
for wing in wings_coordinates_list:
    vstack_stacking = np.vstack((vstack_stacking, wing))
all_wing_points = vstack_stacking[1:, :]
relative_sun_vector_same_shape = np.zeros(np.shape(all_wing_points))
relative_sun_vector_same_shape[:, :3] = n_s
top_all_wing_points = all_wing_points + relative_sun_vector_same_shape * 2
bottom_all_wing_points = all_wing_points - relative_sun_vector_same_shape * 2
total_hull = np.vstack((top_all_wing_points, bottom_all_wing_points))
hull = Delaunay(total_hull)
#plt.plot(total_hull[:,0], total_hull[:,1], 'o')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(total_hull[:, 0], total_hull[:, 1], total_hull[:, 2])
ax.scatter(all_wing_points[:, 0], all_wing_points[:, 1], all_wing_points[:, 2])
for simplex in hull.simplices:
    plt.plot(total_hull[simplex, 0], total_hull[simplex, 1], total_hull[simplex, 2], 'k-')


# Mesh the vane edges for shadow determination
all_meshed_points = np.array([[0, 0, 0]])
for i in range(1, np.shape(vanes_coordinates)[0]+1):
    if (i == np.shape(vanes_coordinates)[0]):
        delta_vec = vanes_coordinates[0, :] - vanes_coordinates[i - 1, :]
    else:
        delta_vec = vanes_coordinates[i, :] - vanes_coordinates[i - 1, :]
    meshed_points = np.zeros((6, 3))

    for j in range(6):
        meshed_points[j, :] = vanes_coordinates[i-1, :] + j * delta_vec/5
    all_meshed_points = np.vstack((all_meshed_points, meshed_points))
meshed_vane_coordinates = all_meshed_points
def vane_torque_body_frame(alpha_1, alpha_2, n_s):
    rotated_points_body_frame = vane_dynamical_model([np.rad2deg(alpha_1)],
                         [np.rad2deg(alpha_2)],
                         1,
                         [vanes_origin],
                         [meshed_vane_coordinates],
                         [vanes_rotation_matrices_list[0]])[0]

    centroid_body_frame, A_vane, n = compute_panel_geometrical_properties(rotated_points_body_frame)
    c_theta = np.dot(n, n_s)

    # Get the vane torque according to the optical model
    if (c_theta >= 0):  # the front is exposed
        f = (W * A_vane * abs(c_theta) / c) * ((
            alpha_front * absorption_reemission_ratio - 2 * rho_s_front * c_theta - rho_d_front * B_front) * n + (
                        alpha_front + rho_d_front) * n_s)
    else:
        f = (W * A_vane * abs(c_theta) / c) * ((
            alpha_back * absorption_reemission_ratio - 2 * rho_s_back * c_theta + rho_d_back * B_back) * n + (
                        alpha_front + rho_d_front) * n_s)

    force_on_vane_body_reference_frame = np.dot(R_BV, f)
    torque_on_body_from_vane = np.cross(centroid_body_frame, force_on_vane_body_reference_frame)

    shadow_bool = vane_shadow(rotated_points_body_frame[2:-1, :], hull)    #  In practice the hull would be updated at each iteration of the propagation
    if (shadow_bool):
        result = np.array([1e23, 1e23, 1e23])
    else:
        result = torque_on_body_from_vane
    return result

def vane_shadow(vane_coordinates_body_fixed, sail_shadow_zone):
    in_hull(vane_coordinates_body_fixed, sail_shadow_zone)
    return any(in_hull(vane_coordinates_body_fixed, sail_shadow_zone))

class vaneAnglesAllocation:
    def __init__(self, target_torque, n_s):
        self.target_torque = target_torque
        self.sun_direction_body_frame = n_s
    def fitness(self, x):
        return (1/3) * np.sum(((vane_torque_body_frame(x[0], x[1], n_s) - target_torque)/1e-6)**2)
    def get_bounds(selfself):
        return [[-np.pi, np.pi], [-np.pi, np.pi]]






res = vane_torque_body_frame(np.deg2rad(23), np.deg2rad(67), n_s)

set_axes_equal(ax)


target_torque = np.array([-1e-6, 0, -0.5 * 1e-6]) # Could make use of that for cases with a different moment arm and the possibility to do torques around all axes
cost_function = lambda p, n_sun=n_s: (1/3) * np.sum(((vane_torque_body_frame(p[0], p[1], n_sun) - target_torque)/1e-6)**2)
t0 = time()
for j in range(10000):
    cost_function([0, 0])
print(time()-t0)

print(cost_function([0, 0]))
t0 = time()
res = minimize(cost_function, (0, 0), method='Nelder-Mead', bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], tol=1e-3)
t1 = time()
print(np.rad2deg(np.array(res.x)), res.nfev)
print(vane_torque_body_frame(res.x[0], res.x[1], n_s))
print(res.message)
print(t1-t0)

t0 = time()
res_2 = direct(cost_function, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], len_tol=(1e-3))
t1 = time()
print(np.rad2deg(res_2.x), res_2.fun, res_2.nfev)
print(vane_torque_body_frame(res_2.x[0], res_2.x[1], n_s))
print(t1-t0)

POTATO_PLOT = True
if (POTATO_PLOT):
    t0 = time()
    alpha_1_range = np.linspace(-np.pi, np.pi, 250)
    alpha_2_range = np.linspace(-np.pi, np.pi, 250)
    vstackT_ = np.zeros(np.shape(alpha_2_range))
    vstackTx_ = np.zeros(np.shape(alpha_2_range))
    vstackTy_ = np.zeros(np.shape(alpha_2_range))
    vstackTz_ = np.zeros(np.shape(alpha_2_range))
    vstackC_ =  np.zeros(np.shape(alpha_2_range))
    for alpha_1 in alpha_1_range:
        hstackT_ = np.array([0])
        hstackTx_ = np.array([0])
        hstackTy_ = np.array([0])
        hstackTz_ = np.array([0])
        hstackC_ = np.array([0])
        for alpha_2 in alpha_2_range:
            T = vane_torque_body_frame(alpha_1, alpha_2, n_s)
            C = np.sum(T**2) * (1/3)
            Tx = T[0]
            Ty = T[1]
            Tz = T[2]
            hstackT_ = np.hstack((hstackT_, np.array([np.linalg.norm(T)])))
            hstackTx_ = np.hstack((hstackTx_, np.array([Tx])))
            hstackTy_ = np.hstack((hstackTy_, np.array([Ty])))
            hstackTz_ = np.hstack((hstackTz_, np.array([Tz])))
            hstackC_ = np.hstack((hstackC_, np.array([C])))
        vstackT_ = np.vstack((vstackT_, hstackT_[1:]))
        vstackTx_ = np.vstack((vstackTx_, hstackTx_[1:]))
        vstackTy_ = np.vstack((vstackTy_, hstackTy_[1:]))
        vstackTz_ = np.vstack((vstackTz_, hstackTz_[1:]))
        vstackC_ = np.vstack((vstackC_, hstackC_[1:]))
    print(time()-t0)
    ZT = vstackT_[1:, :]
    BT = vstackT_[1:, :]
    BT[BT < 1e20] = 0
    BT[BT > 1e20] = 1
    ZT[ZT > 1e20] = None
    ZTx = vstackTx_[1:, :]
    ZTx[ZTx > 1e20] = None
    ZTy = vstackTy_[1:, :]
    ZTy[ZTy > 1e20] = None
    ZTz = vstackTz_[1:, :]
    ZTz[ZTz > 1e20] = None
    ZC = vstackC_[1:, :]
    ZC[ZC > 1e20] = None
    alpha_1_range, alpha_2_range = np.meshgrid(alpha_1_range, alpha_2_range)

    PLOT_SURFACE = True
    if PLOT_SURFACE:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(np.rad2deg(alpha_2_range), np.rad2deg(alpha_1_range), BT, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel("alpha_1")
        ax.set_ylabel("alpha_2")

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(np.rad2deg(alpha_2_range), np.rad2deg(alpha_1_range), ZC, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel("alpha_1")
        ax.set_ylabel("alpha_2")

    plt.figure()
    flattened_Tx = ZTx.reshape(-1)
    flattened_Tz = ZTz.reshape(-1)
    plt.scatter(flattened_Tx, flattened_Tz, s=1)
    plt.xlabel("Tx")
    plt.ylabel("Tz")
    plt.show()

