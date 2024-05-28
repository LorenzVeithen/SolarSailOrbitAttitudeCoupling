# Considering a single vane for a given torque, let's determine the right angles of the vanes
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
import pygmo as pg
from scipy.optimize import minimize, direct

from constants import *
from MiscFunctions import *

from ACS_dynamicalModels import vane_dynamical_model
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from ACS_optimisationProblems import vaneAnglesAllocationProblem


sun_angle_alpha = np.deg2rad(125.264)
sun_angle_beta = np.deg2rad(45)
n_s = np.array([np.sin(sun_angle_alpha) * np.cos(sun_angle_beta), np.sin(sun_angle_alpha) * np.sin(sun_angle_beta), -np.cos(sun_angle_alpha)])   # In the body reference frame
#n_s = np.array([1, 1, 1])
n_s = n_s/np.linalg.norm(n_s)


# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("vanes", boom_list)
acs_object.set_vane_characteristics(vanes_coordinates_list, vanes_origin_list, vanes_rotation_matrices_list, 0,
                                    np.array([0, 0, 0]), 0.0045, vanes_rotational_dof)

sail = sail_craft("ACS3",
                  len(wings_coordinates_list),
                  len(vanes_coordinates_list),
                  wings_coordinates_list,
                  vanes_coordinates_list,
                  wings_optical_properties,
                  vanes_optical_properties,
                  sail_I,
                  sail_mass,
                  sail_mass_without_wings,
                  sail_nominal_CoM,
                  sail_material_areal_density,
                  sail_material_areal_density,
                  acs_object)

vaneAngleProblem = vaneAnglesAllocationProblem(0,
                                               ([-np.pi, -np.pi, 0], [np.pi, np.pi, 1]),
                                               10,
                                               sail,
                                               acs_object,
                                               include_shadow=True)

vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([-5e-6, 0, -3e-6]), n_s, vane_variable_optical_properties=True)   # and the next time you can put False
res = vaneAngleProblem.fitness([np.deg2rad(23), np.deg2rad(67), 1])

t0 = time()
res_2 = direct(vaneAngleProblem.fitness, bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (0, 1)], len_tol=(1e-3))
t1 = time()
print(np.rad2deg(res_2.x[:2]), res_2.x[2], res_2.fun, res_2.nfev)
print(vaneAngleProblem.single_vane_torque([res_2.x[0], res_2.x[1]]))
print(t1-t0)


POTATO_PLOT = True
if (POTATO_PLOT):
    t0 = time()
    alpha_1_range = np.linspace(-np.pi, np.pi, 100)
    alpha_2_range = np.linspace(-np.pi, np.pi, 100)
    vstackT_ = np.zeros(np.shape(alpha_2_range))
    vstackTx_ = np.zeros(np.shape(alpha_2_range))
    vstackTy_ = np.zeros(np.shape(alpha_2_range))
    vstackTz_ = np.zeros(np.shape(alpha_2_range))
    vstackC_ = np.zeros(np.shape(alpha_2_range))
    for alpha_1 in alpha_1_range:
        hstackT_ = np.array([0])
        hstackTx_ = np.array([0])
        hstackTy_ = np.array([0])
        hstackTz_ = np.array([0])
        hstackC_ = np.array([0])
        for alpha_2 in alpha_2_range:
            T = vaneAngleProblem.single_vane_torque([alpha_1, alpha_2])
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


TEST_COST_FUNCTION = False
if (TEST_COST_FUNCTION):
    t0 = time()
    for j in range(10000):
        vaneAngleProblem.single_vane_torque([0, 0])
    print(time()-t0)

TEST_OPTIMISATION_ALGORITHMS = False
if (TEST_OPTIMISATION_ALGORITHMS):
    t0 = time()
    print(vaneAngleProblem.fitness([0, 0])[0])
    res = minimize(vaneAngleProblem.fitness, (-1.5, -1.5), method='Nelder-Mead', bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (0, 1)], tol=1e-4)
    t1 = time()
    print(np.rad2deg(np.array(res.x)), res.nfev)
    print(vaneAngleProblem.single_vane_torque([res.x[0], res.x[1]]))
    print(t1-t0)

    t0 = time()
    res_2 = direct(vaneAngleProblem.fitness, bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (0, 1)], len_tol=(1e-3))
    t1 = time()
    print(np.rad2deg(res_2.x), res_2.fun, res_2.nfev)
    print(vaneAngleProblem.single_vane_torque([res_2.x[0], res_2.x[1]]))
    print(t1-t0)

    probVA = pg.problem(vaneAngleProblem)
    #print(probVA)
    pop = pg.population(probVA, size = 15)
    algo = pg.algorithm(pg.de(gen = 7))
    algo.set_verbosity(1)
    t0 = time()
    pop = algo.evolve(pop)
    t1 = time()
    best_fitness = pop.get_f()[pop.best_idx()]
    best_individual = pop.champion_x
    print(best_fitness, np.rad2deg(best_individual))
    print(probVA.get_fevals())

    print(vaneAngleProblem.single_vane_torque(best_individual))
    print(pop.problem.get_fevals())
    print(t1-t0)

    # Direct seems to be the best, always converges, is deterministic and not too expensive either, such a great pic
    # Direct is generally good for low dimensionality (here only 2 so we should be good
