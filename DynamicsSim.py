# Load standard modules
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import *
from controllers import *

import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from MiscFunctions import set_axes_equal, axisRotation


# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

class sailCoupledDynamicsProblem(sail_craft):

    def __init__(self,
                 num_panels,
                 nominal_sail_mass,
                 initial_CoM_position_body_frame,
                 initial_inertia_tensor_body_frame,
                 initial_panels_coordinates_body_frame,
                 initial_panels_optical_properties,
                 ACS_system,
                 initial_state,  # 1 x 12 vector [translational, rotational]
                 simulation_start_epoch,
                 simulation_end_epoch):
        super(sail_craft).__init__(num_panels, nominal_sail_mass, initial_CoM_position_body_frame, initial_inertia_tensor_body_frame, initial_panels_coordinates_body_frame, initial_panels_optical_properties, ACS_system)
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch
        self.sail_num_panels = num_panels

        # Load spice kernels
        spice.load_standard_kernels()

        # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
        self.global_frame_origin = "Earth"
        self.global_frame_orientation = "J2000"

        # Define bodies that are propagated
        self.bodies_to_create = ["Earth", "Sun"]
        self.bodies_to_propagate = ["ACS3"]
        self.central_bodies = ["Earth"]

        self.occulting_bodies_dict = dict()
        self.occulting_bodies_dict["Sun"] = ["Earth"]

        # integrator settings
        self.initial_step_size = 1.0
        self.control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12)
        self.validation_settings = propagation_setup.integrator.step_size_validation(1E-5, 1E3)
        self.initial_state = initial_state

    def define_simulation_bodies(self):
        body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)
        # Add vehicle object to system of bodies
        body_settings.add_empty_settings("ACS3")

        # Setup that the sail_craft body functions are passed
        body_settings.get("ACS3").rigid_body_settings = environment_setup.rigid_body.custom_time_dependent_rigid_body_properties(
            mass_function=sail_craft.get_sail_mass,
            center_of_mass_function=sail_craft.get_sail_center_of_mass,
            inertia_tensor_function=sail_craft.get_sail_inertia_tensor)

        # Written such that it can be extended in the future
        # Compile the panel properties
        panels_settings = []
        for j in range(2):
            if (j == 0):
                panel_type_str = "Sail"
                number_of_panels = self.sail_num_panels
            else:
                panel_type_str = "Vane"
                number_of_panels = self.sail_num_vanes

            for i in range(number_of_panels):
                ith_panel_normal = lambda t: sail_craft.get_ith_panel_surface_normal(self, t, panel_id=i, panel_type=panel_type_str)
                ith_panel_area = lambda t: sail_craft.get_ith_panel_area(self, t, panel_id=i, panel_type=panel_type_str)
                ith_panel_centroid = lambda t: sail_craft.get_ith_panel_centroid(self, t, panel_id=i, panel_type=panel_type_str)
                ith_panel_properties_list = [lambda t: sail_craft.get_ith_panel_optical_properties(self, t, panel_id=i, panel_type=panel_type_str)[j] for j in range(10)]
                panel_geom = environment_setup.vehicle_systems.time_varying_panel_geometry(surface_normal=ith_panel_normal,
                                                                                            area=single_panel_area,
                                                                                            frame_orientation="VehicleFixed")

                reflection_law = environment_setup.radiation_pressure.solar_sail_optical_body_panel_reflection(1, 1, 0, 0, 0, 0,
                                                                                                               0, 0, 0, 0)
                panel = environment_setup.vehicle_systems.body_panel_settings(panel_type_id=panel_type_str,
                                                                               panel_reflection_law=reflection_law,
                                                                               panel_geometry=panel_geom)
                panels_settings.append(panel)

        panelled_body = environment_setup.vehicle_systems.full_panelled_body_settings(
                panel_settings=panels_settings)  # No rotational models for the panels
        body_settings.get('ACS3').vehicle_shape_settings = panelled_body

        vehicle_target_settings = environment_setup.radiation_pressure.panelled_radiation_target(self.occulting_bodies_dict)

        constant_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # create rotation model settings and assign to body settings of "ACS3"
        body_settings.get("ACS3").rotation_model_settings = environment_setup.rotation_model.constant_rotation_model(
            self.global_frame_orientation,
            "VehicleFixed",
            constant_orientation)

        # Create system of bodies (in this case only Earth)
        bodies = environment_setup.create_system_of_bodies(body_settings)
        sail_craft.setBodies(self, bodies)
        return bodies, vehicle_target_settings

    def define_dependent_variables(self):
        # DEPENDENT VARIABLES
        return [propagation_setup.dependent_variable.keplerian_state('ACS3', 'Earth'),
                propagation_setup.dependent_variable.received_irradiance_shadow_function("ACS3", "Sun"),
                propagation_setup.dependent_variable.single_acceleration(
                    propagation_setup.acceleration.radiation_pressure_type, "ACS3", "Sun"),
                propagation_setup.dependent_variable.relative_position("ACS3", "Sun"),
                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Earth', 'ACS3'),
                propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('ACS3')]

    def define_numerical_environment(self):
        # Create termination settings
        termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch,
                                                                             terminate_exactly_on_final_condition=True)

        # Create numerical integrator settings
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step=self.initial_step_size,
            coefficient_set=propagation_setup.integrator.rkf_45,
            step_size_control_settings=self.control_settings,
            step_size_validation_settings=self.validation_settings)
        return termination_settings, integrator_settings

    def define_dynamical_environment(self, bodies, vehicle_target_settings):
        environment_setup.add_radiation_pressure_target_model(
            bodies, "ACS3", vehicle_target_settings)

        # Translational dynamics settings
        acceleration_settings_acs3 = dict(
            Earth=[propagation_setup.acceleration.point_mass_gravity()],
            Sun=[propagation_setup.acceleration.radiation_pressure()]
        )

        acceleration_settings = {"ACS3": acceleration_settings_acs3}
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, self.bodies_to_propagate, self.central_bodies)

        # Rotational dynamics settings
        torque_settings_on_sail = dict()  # dict(Earth = [propagation_setup.torque.second_degree_gravitational()])
        torque_settings = {'ACS3': torque_settings_on_sail}
        torque_models = propagation_setup.create_torque_models(bodies, torque_settings, self.bodies_to_propagate)
        return acceleration_models, torque_models

    def define_propagators(self, integrator_settings, acceleration_models, torque_models, dependent_variables, termination_settings):
        # Create propagation settings
        translational_propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            acceleration_models,
            self.bodies_to_propagate,
            self.initial_state[:6],
            self.simulation_start_epoch,
            integrator_settings,
            termination_settings,
            propagation_setup.propagator.gauss_modified_equinoctial)

        rotational_propagator_settings = propagation_setup.propagator.rotational(torque_models,
                                                                                 self.bodies_to_propagate,
                                                                                 self.initial_state[6:],
                                                                                 self.simulation_start_epoch,
                                                                                 integrator_settings,
                                                                                 termination_settings)

        # MULTI-TYPE PROPAGATOR
        propagator_list = [translational_propagator_settings, rotational_propagator_settings]
        combined_propagator_settings = propagation_setup.propagator.multitype(propagator_list,
                                                                              integrator_settings,
                                                                              self.simulation_start_epoch,
                                                                              termination_settings,
                                                                              output_variables=dependent_variables)
        return combined_propagator_settings

    def run_sim(self, bodies, combined_propagator_settings):
        simulator = numerical_simulation.create_dynamics_simulator(bodies, combined_propagator_settings)
        state_history = simulator.state_history
        dependent_variable_history = simulator.dependent_variable_history

        # Extract the resulting state history and convert it to a ndarray
        states_array = result2array(state_history)
        dependent_variable_array = result2array(dependent_variable_history)
        return states_array, dependent_variable_array

    def write_results_to_fiel(self, states_array, dependent_variable_array):
        pass
