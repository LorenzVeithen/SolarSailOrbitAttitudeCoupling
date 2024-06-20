import numpy as np
import pygmo as pg
from constants import *
from MiscFunctions import *
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from time import time
from vaneControllerMethods import vaneTorqueAllocationProblem, buildEllipseCoefficientFunctions, ellipseCoefficientFunction
from vaneControllerMethods import cart_to_pol, vaneAnglesAllocationProblem, vaneAngleAllocationScaling
import matplotlib.pyplot as plt
from scipy.optimize import direct, golden

include_shadow = False
solar_irradiance = 1400
alpha_s_deg = -139
beta_s_deg = 98

torque_allocation_problem_objective_function_weights = [1, 0]
target_torque = np.array([0.0, 0., 5.])
previous_torque_allocation_solution = np.array([0] * 12)

# tolerances
tol_vane_angle_determination_start_golden_section = algorithm_constants["vane_angle_determination_start_golden_section"]
tol_vane_angle_determination_golden_section = algorithm_constants["vane_angle_determination_golden_section"]
tol_vane_angle_determination_global_search = algorithm_constants["vane_angle_determination_global_search"]

tol_torque_allocation_problem_constraint = algorithm_constants["torque_allocation_problem_constraint"]
tol_torque_allocation_problem_objective = algorithm_constants["torque_allocation_problem_objective"]
tol_torque_allocation_problem_x = algorithm_constants["torque_allocation_problem_x"]

# sunlight vector
sunlight_vector_body_frame = np.array([np.sin(np.deg2rad(alpha_s_deg)) * np.cos(np.deg2rad(beta_s_deg)),
                            np.sin(np.deg2rad(alpha_s_deg)) * np.sin(np.deg2rad(beta_s_deg)),
                            -np.cos(np.deg2rad(alpha_s_deg))])   # In the body reference frame
sunlight_vector_body_frame = sunlight_vector_body_frame/np.linalg.norm(sunlight_vector_body_frame)

# attitude control system object
acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    vane_has_ideal_model,
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits,
                                    vanes_optical_properties)

# sail object
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

# feasible torque ellipse coefficients
ellipse_coefficient_functions_list = []
for i in range(6):
    filename = f'/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/AMS/Datasets/Ideal_model/vane_1/dominantFitTerms/{["A", "B", "C", "D", "E", "F"][i]}_shadow_{str(include_shadow)}.txt'
    built_function = buildEllipseCoefficientFunctions(filename)
    ellipse_coefficient_functions_list.append(
        lambda aps, bes, f=built_function: ellipseCoefficientFunction(aps, bes, f))

# vane torque allocation problem
tap = vaneTorqueAllocationProblem(acs_object,
                                  wings_coordinates_list,
                                  True,
                                  include_shadow,
                                  ellipse_coefficient_functions_list,
                                  vanes_optical_properties,
                                  w1=torque_allocation_problem_objective_function_weights[0],
                                  w2=torque_allocation_problem_objective_function_weights[1],
                                  num_shadow_mesh_nodes=10)

tap.set_desired_torque(target_torque, previous_torque_allocation_solution)
t0 = time()
tap.set_attaignable_moment_set_ellipses(sunlight_vector_body_frame)
prob = pg.problem(tap)
nl = pg.nlopt('cobyla')
nl.xtol_rel = tol_torque_allocation_problem_x
nl.ftol_rel = tol_torque_allocation_problem_objective
algo = pg.algorithm(uda=nl)
algo.set_verbosity(1)
pop = pg.population(prob, size=1, seed=42)
#pop.push_back(x=previous_torque_allocation_solution)  # TODO: Use the previous solution as initial guess to make the optimisation faster.
pop.problem.c_tol = tol_torque_allocation_problem_constraint
pop = algo.evolve(pop)

x_final = pop.champion_x
final_torques = x_final.reshape((acs_object.number_of_vanes, 3))
resulting_sail_torque = final_torques.sum(axis=0)

# Determine associated angles
torque_from_vane_angles = []
for vane_id in range(acs_object.number_of_vanes):
    optimised_vane_torque = final_torques[vane_id, :]
    # determine the required vane angles for this
    vaneAngleProblem = tap.vane_angle_problem_objects_list[vane_id]
    vane_angle_determination_bounds = [(vaneAngleProblem.get_bounds()[0][i], vaneAngleProblem.get_bounds()[1][i]) for i in
                                       range(len(vaneAngleProblem.get_bounds()[0]))]
    vane_angle_allocation_results = vaneAngleAllocationScaling(1, optimised_vane_torque / tap.scaling_list[vane_id],
                                                               sunlight_vector_body_frame,
                                                               vaneAngleProblem,
                                                               vane_angle_determination_bounds,
                                                               tol_vane_angle_determination_global_search,
                                                               tol_vane_angle_determination_start_golden_section)[1]
    if (vane_angle_allocation_results.fun > tol_vane_angle_determination_start_golden_section):
        f_golden = lambda t, Td=optimised_vane_torque / tap.scaling_list[vane_id], \
                          n_s=sunlight_vector_body_frame, vaneP=vaneAngleProblem, \
                          vane_bounds=vane_angle_determination_bounds, \
                          tol_global=tol_vane_angle_determination_global_search,\
                          tol_golden=tol_vane_angle_determination_start_golden_section: \
                        vaneAngleAllocationScaling(t, Td, n_s, vaneP, vane_bounds, tol_global, tol_golden)[0]
        minimizer = golden(f_golden, brack=(0, 1), tol=tol_vane_angle_determination_golden_section)
        vane_angle_allocation_results = vaneAngleAllocationScaling(minimizer, optimised_vane_torque / tap.scaling_list[vane_id],
                                                                   sunlight_vector_body_frame,
                                                                   vaneAngleProblem,
                                                                   vane_angle_determination_bounds,
                                                                   tol_vane_angle_determination_global_search,
                                                                   tol_vane_angle_determination_start_golden_section)[1]
        print(f'scaling factor to fit in feasibility={minimizer}')
    torque_from_vane_angles.append(vaneAngleProblem.single_vane_torque([vane_angle_allocation_results.x[0],
                                                                   vane_angle_allocation_results.x[1]]) * tap.scaling_list[vane_id])
    print(f"vane angles: {np.rad2deg(vane_angle_allocation_results.x[:2])}")
    print(f"requested torque={optimised_vane_torque}")
    print(f"obtained torque={torque_from_vane_angles[vane_id]}")
    print(f"vane angle determination objective function value={vane_angle_allocation_results.fun}")
t1 = time()

print(f'Number of function evaluations: {pop.problem.get_fevals()}')
print(f'Number of gradient evaluations: {pop.problem.get_gevals()}')
print(final_torques)
print(resulting_sail_torque)

s = np.array([0., 0., 0.])
for vane_id in range(acs_object.number_of_vanes):
    s += torque_from_vane_angles[vane_id]



print(f'requested sail torque direction= {target_torque/np.linalg.norm(target_torque)}')
print(f'resulting sail torque direction from angles= {s/np.linalg.norm(s)}')
print(f"full computing time: {t1-t0}")


for vane_id in range(acs_object.number_of_vanes):
    print(vane_id)
    vaneAngleProblem = tap.vane_angle_problem_objects_list[vane_id]
    optimised_vane_torque = final_torques[vane_id, :]
    # Check that the selected thrust is indeed in the feasibility region
    alpha_1_range = np.linspace(-np.pi, np.pi, 181)
    alpha_2_range = np.linspace(-np.pi, np.pi, 181)
    T_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    Tx_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    Ty_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    Tz_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    C_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    color2 = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    color1 = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    for i, alpha_1 in enumerate(alpha_1_range):
        for j, alpha_2 in enumerate(alpha_2_range):
            T = vaneAngleProblem.single_vane_torque([alpha_1, alpha_2]) * tap.scaling_list[vane_id]
            T_[i, j] = np.linalg.norm(T)
            Tx_[i, j] = T[0]
            Ty_[i, j] = T[1]
            Tz_[i, j] = T[2]
            C_[i, j] = np.sum(T ** 2) * (1 / 3)
            color2[i, j] = 1 if (abs(alpha_2) < 1e-15) else 0
            color1[i, j] = 1 if (abs(alpha_1) < 1e-15) else 0
    ZT = T_
    BT = T_
    BT[BT < 1e20] = 0
    BT[BT > 1e20] = 1
    ZT[ZT > 1e20] = None
    ZTx = Tx_
    ZTx[ZTx > 1e20] = None
    ZTy = Ty_
    ZTy[ZTy > 1e20] = None
    ZTz = Tz_
    ZTz[ZTz > 1e20] = None
    ZC = C_
    ZC[ZC > 1e20] = None
    alpha_1_range_grid, alpha_2_range_grid = np.meshgrid(alpha_1_range, alpha_2_range)

    flattened_Tx = ZTx.reshape(-1)
    flattened_Ty = ZTy.reshape(-1)
    flattened_Tz = ZTz.reshape(-1)
    flattened_color2 = color2.reshape(-1)
    flattened_color1 = color1.reshape(-1)
    T00 = vaneAngleProblem.single_vane_torque([np.deg2rad(0), np.deg2rad(0)]) * tap.scaling_list[vane_id]

    current_torque_from_vane_angles = torque_from_vane_angles[vane_id]
    plt.figure()
    plt.title(f"Vane ID: {vane_id} - TyTz")
    plt.scatter(flattened_Ty, flattened_Tz, s=1, label="all points")
    plt.plot(flattened_Ty[flattened_color2==1], flattened_Tz[flattened_color2==1], color='r', label="alpha_2=0")
    plt.plot(flattened_Ty[flattened_color1==1], flattened_Tz[flattened_color1==1], color='g', label="alpha_1=0")
    plt.scatter(T00[1], T00[2], s=50, label="alpha_1=alpha2=0")
    plt.scatter(optimised_vane_torque[1], optimised_vane_torque[2], label="optimised vane torque")
    plt.scatter(current_torque_from_vane_angles[1], current_torque_from_vane_angles[2], label="vane angles torque")
    plt.xlabel("Ty")
    plt.ylabel("Tz")
    plt.legend()


    plt.figure()
    plt.title(f"Vane ID: {vane_id} - TxTz")
    plt.scatter(flattened_Tx, flattened_Tz, s=1, label="all points")
    plt.plot(flattened_Tx[flattened_color2==1], flattened_Tz[flattened_color2==1], color='r', label="alpha_2=0")
    plt.plot(flattened_Tx[flattened_color1==1], flattened_Tz[flattened_color1==1], color='g', label="alpha_1=0")
    plt.scatter(T00[0], T00[2], s=50, label="alpha_1=alpha2=0")
    plt.scatter(optimised_vane_torque[0], optimised_vane_torque[2], label="optimised vane torque")
    plt.scatter(current_torque_from_vane_angles[0], current_torque_from_vane_angles[2], label="vane angles torque")
    plt.xlabel("Tx")
    plt.ylabel("Tz")
    plt.legend()
plt.show()
