import numpy as np
import pygmo as pg
from constants import *
from MiscFunctions import *
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from time import time
from vaneControllerMethods import vaneTorqueAllocationProblem, buildEllipseCoefficientFunctions, ellipseCoefficientFunction

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


tap = vaneTorqueAllocationProblem(acs_object, sail, True, include_shadow, ellipse_coefficient_functions_list)
tap.set_desired_torque(np.array([-0.01, 0.1, -5]), np.array([0, 0, 0]))
prob = pg.problem(tap)
algo = pg.algorithm(uda = pg.nlopt('auglag'))
algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
algo.set_verbosity(100) # in this case this correspond to logs each 200 objevals
t0 = time()
pop = pg.population(prob, size = 1)
t1 = time()
pop = algo.evolve(pop)

print(pop.problem.get_fevals())
print(pop.problem.get_gevals())
print(pop.champion_x.reshape((4, 3)))
print(pop.champion_x.reshape((4, 3)).sum(axis=0))
print(t1-t0)