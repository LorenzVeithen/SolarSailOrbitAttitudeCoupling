import numpy as np
import pygmo as pg
from constants import *
from MiscFunctions import *
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from time import time
from vaneControllerMethods import vaneTorqueAllocationProblem, buildEllipseCoefficientFunctions, ellipseCoefficientFunction
from vaneControllerMethods import cart_to_pol



include_shadow = False
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

ellipse_coefficient_functions_list = []
for i in range(6):
    filename = f'/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/AMS/Datasets/Ideal_model/vane_1/dominantFitTerms/{["A", "B", "C", "D", "E", "F"][i]}_shadow_{str(include_shadow)}.txt'
    built_function = buildEllipseCoefficientFunctions(filename)
    ellipse_coefficient_functions_list.append(
        lambda aps, bes, f=built_function: ellipseCoefficientFunction(aps, bes, f))

alpha_s_deg = 100
beta_s_deg = 20

# initialise Jit function
sunlight_vector_body_frame =  np.array([np.sin(np.deg2rad(alpha_s_deg)) * np.cos(np.deg2rad(beta_s_deg)),
                            np.sin(np.deg2rad(alpha_s_deg)) * np.sin(np.deg2rad(beta_s_deg)),
                            -np.cos(np.deg2rad(alpha_s_deg))])   # In the body reference frame
sunlight_vector_body_frame = sunlight_vector_body_frame/np.linalg.norm(sunlight_vector_body_frame)

tap = vaneTorqueAllocationProblem(acs_object, sail, True, include_shadow, ellipse_coefficient_functions_list, w1=1, w2=0)


tap.set_desired_torque(np.array([-2.0, 0., 1]), np.array([0]*12))
t0 = time()
tap.set_attaignable_moment_set_ellipses(sunlight_vector_body_frame, W=1400)
t1 = time()
prob = pg.problem(tap)
nl = pg.nlopt('cobyla')
nl.xtol_rel = 1E-2 # Change the default value of the xtol_rel stopping criterion
algo = pg.algorithm(uda = nl)
#algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
algo.set_verbosity(1)  # in this case this correspond to logs each 200 objevals

pop = pg.population(prob, size=1, seed=42)
pop.problem.c_tol = 1E-3
pop = algo.evolve(pop)

print(pop.problem.get_fevals())
print(pop.problem.get_gevals())
print(pop.champion_x.reshape((acs_object.number_of_vanes, 3)))
print(pop.champion_x.reshape((acs_object.number_of_vanes, 3)).sum(axis=0))
print(t1-t0)

print("time test")
t0 = time()
for i in range(10000):
    prob.fitness(pop.champion_x)
print((time()-t0))