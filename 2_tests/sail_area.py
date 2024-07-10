import numpy as np
from constants import *
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems


# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants, include_shadow=False)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    "double_ideal_optical_model",
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits,
                                    vanes_optical_properties,
                                    torque_allocation_problem_objective_function_weights=[2./3., 1./3.])

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

print(sail.get_ith_panel_area(0, "wing")*4)

