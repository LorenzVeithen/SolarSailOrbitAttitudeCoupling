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
from vaneControllerMethods import vaneAnglesAllocationProblem
from scipy.optimize import shgo, golden

sun_angle_alpha = np.deg2rad(34)
sun_angle_beta = np.deg2rad(-123)
n_s = np.array([np.sin(sun_angle_alpha) * np.cos(sun_angle_beta), np.sin(sun_angle_alpha) * np.sin(sun_angle_beta), -np.cos(sun_angle_alpha)])   # In the body reference frame
n_s = n_s/np.linalg.norm(n_s)


# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("vanes", boom_list)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    vane_has_ideal_model,
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits)

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

vaneAngleProblem = vaneAnglesAllocationProblem(1,
                                               ([-np.pi, -np.pi], [np.pi, np.pi]),
                                               10,
                                               wings_coordinates_list,
                                               acs_object,
                                               include_shadow=False)





print(vaneAngleProblem.single_vane_torque([0, 0]))


vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0., 0.70973164, -0.4875445]), n_s, vane_variable_optical_properties=True, vane_optical_properties_list=vanes_optical_properties)   # and the next time you can put False

fit_func = lambda x:  vaneAngleProblem.fitness(x)[0]
boundss = [(vaneAngleProblem.get_bounds()[0][i], vaneAngleProblem.get_bounds()[1][i]) for i in
           range(len(vaneAngleProblem.get_bounds()[0]))]

def vaneAngleAllocationScaling(t, desired_torque):
    scaled_desired_torque = desired_torque * t
    vaneAngleProblem.update_vane_angle_determination_algorithm(scaled_desired_torque, n_s,
                                                               vane_variable_optical_properties=True, vane_optical_properties_list=vanes_optical_properties)  # and the next time you can put False
    optRes = direct(fit_func, bounds=boundss, len_tol=(1e-4))
    obtainedFitness = vaneAngleProblem.fitness([optRes.x[0], optRes.x[1]])[0]
    if (obtainedFitness < 1e-4):
        return 1
    else:
        return -1

f = lambda t, Td=np.array([0, 0.4*3, 0.3*3]): vaneAngleAllocationScaling(t, Td)

minimizer = golden(f, brack=(0, 1), tol=1e-1)



SELECTED_ALG = True
if (SELECTED_ALG):
    print("DIRECT selected algorithm")
    t0 = time()
    res_2 = direct(fit_func, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], len_tol=(1e-4))
    t1 = time()
    print(np.rad2deg(res_2.x[:2]), res_2.fun, res_2.nfev)
    print(vaneAngleProblem.single_vane_torque([res_2.x[0], res_2.x[1]]))
    #print(vaneAngleProblem.fitness([res_2.x[0], res_2.x[1]])[0])
    print(f'time: {t1-t0}')


TEST_OPTIMISATION_ALGORITHMS = True
if (TEST_OPTIMISATION_ALGORITHMS):
    print("Nelder-Mead")
    initial_guess = tuple(np.deg2rad([156.1042524, 50.86419753]))
    t0 = time()
    res = minimize(vaneAngleProblem.fitness, initial_guess, method='Nelder-Mead', bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], tol=1e-4)
    t1 = time()
    print(np.rad2deg(np.array(res.x)), res.nfev)
    print(vaneAngleProblem.single_vane_torque([res.x[0], res.x[1]]))
    print(t1-t0)

    print("DIRECT")
    t0 = time()
    res_2 = direct(vaneAngleProblem.fitness, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], len_tol=(1e-4))
    t1 = time()
    print(np.rad2deg(res_2.x), res_2.fun, res_2.nfev)
    print(vaneAngleProblem.single_vane_torque([res_2.x[0], res_2.x[1]]))
    print(t1-t0)

    """
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
    """

    # Direct seems to be the best, always converges, is deterministic and not too expensive either, such a great pic
    # Direct is generally good for low dimensionality (here only 2 so we should be good
