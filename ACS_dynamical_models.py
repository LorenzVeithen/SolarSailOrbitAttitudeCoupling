"""
Functions describing the attitude control systems dynamics.
This code was outsourced from attitudeControllersClass.py to allow Numba optimisation
"""


from scipy.spatial.transform import Rotation as R
import numpy as np
from MiscFunctions import all_equal, closest_point_on_a_segment_to_a_third_point, compute_panel_geometrical_properties
from numba import jit

@jit(nopython=True, cache=True)
def vane_dynamical_model(rotation_x_deg,
                         rotation_y_deg,
                         number_of_vanes,
                         vane_reference_frame_origin_list,
                         vane_panels_coordinates_list,
                         vane_reference_frame_rotation_matrix_list):
    new_vane_coordinates = []
    for i in range(number_of_vanes):  # For each vane
        current_vane_origin = vane_reference_frame_origin_list[i]
        current_vane_coordinates = vane_panels_coordinates_list[i]
        current_vane_frame_rotation_matrix = vane_reference_frame_rotation_matrix_list[i]

        rotated_vane_coordinates = np.zeros(np.shape(current_vane_coordinates))
        for j in range(len(current_vane_coordinates[:, 0])):  # For each coordinate of the panel
            # Get the panel coordinate points in the vane-centered coordinate system
            current_vane_coordinate_vane_reference_frame = np.matmul(np.linalg.inv(current_vane_frame_rotation_matrix),
                                                                     current_vane_coordinates[j,
                                                                     :] - current_vane_origin)
            # Now rotate along the vane-fixed x-axis and then y-axis
            Rx = R.from_euler('x', rotation_x_deg[i], degrees=True).as_matrix()
            Ry = R.from_euler('y', rotation_y_deg[i], degrees=True).as_matrix()
            vane_rotation_matrix = np.matmul(Ry, Rx)
            current_vane_coordinate_rotated_vane_reference_frame = np.matmul(vane_rotation_matrix,
                                                                             current_vane_coordinate_vane_reference_frame)

            # Convert back to the body fixed reference frame
            current_vane_coordinate_rotated_body_fixed_reference_frame = np.matmul(current_vane_frame_rotation_matrix,
                                                                                   current_vane_coordinate_rotated_vane_reference_frame) + current_vane_origin
            rotated_vane_coordinates[j, :] = current_vane_coordinate_rotated_body_fixed_reference_frame
        new_vane_coordinates.append(rotated_vane_coordinates)
    return new_vane_coordinates

@jit(nopython=True, cache=True)
def shifted_panel_dynamical_model(wings_shifts_list,
                                  number_of_wings,
                                  wings_coordinates_list,
                                  wings_reference_frame_rotation_matrix_list,
                                  retain_wings_area_bool,
                                  point_to_boom_belonging_list,
                                  max_wings_inwards_translations_list,
                                  booms_coordinates_list):
    wing_coordinates_list = []
    for i in range(number_of_wings):
        current_wing_coordinates = wings_coordinates_list[i]
        current_wing_shifts = wings_shifts_list[i]
        new_current_panel_coordinates = np.zeros(np.shape(current_wing_coordinates))
        current_wing_reference_frame_rotation_matrix = wings_reference_frame_rotation_matrix_list[i]
        if (not retain_wings_area_bool): current_wing_boom_belongings = point_to_boom_belonging_list[i]
        for j, point in enumerate(current_wing_coordinates[:, :3]):
            if (retain_wings_area_bool):
                # Here, the panel is just shifted without any shape deformation. The shift is made along the Y-axis of the considered quadrant
                # The tether-spool system dictates the movement
                if (not all_equal(current_wing_shifts)):
                    raise Exception(
                        "Inconsistent inputs for the shifted panels with constant area. All shifts need to be equal.")
                elif (current_wing_shifts[0] < max_wings_inwards_translations_list[i]):
                    raise Exception(
                        "Requested shift is larger than allowable by the define geometry or positive (only negative are permitted). "
                        + f"requested shift: {current_wing_shifts[0]}, maximum negative shift: {max_wings_inwards_translations_list[i]}")
                else:
                    # Rotate to the quadrant reference frame
                    point_coordinates_wing_reference_frame = np.matmul(
                        np.linalg.inv(current_wing_reference_frame_rotation_matrix),
                        point)  # Get the position vector in the wing reference frame
                    translated_point_coordinates_wing_reference_frame = point_coordinates_wing_reference_frame + \
                                                                        current_wing_shifts[j] * np.array(
                        [0, 1, 0])  # Get the translated point in the wing reference frame
                    new_point_coordinates_body_fixed_frame = np.matmul(current_wing_reference_frame_rotation_matrix,
                                                                       translated_point_coordinates_wing_reference_frame)  # Rotate back to body fixed reference frame
            else:
                # Just shift the panels according to the inputs assuming that the material is extensible enough (simplifying assumption to avoid melting one's brain)
                # More general implementation but less realistic implementation for most cases
                related_boom = current_wing_boom_belongings[j]
                if (related_boom != None):  # Only do a shift if the attachment point belongs to a boom, not otherwise
                    boom_vector = booms_coordinates_list[related_boom][1, :] - booms_coordinates_list[
                                                                                        related_boom][0, :]
                    boom_vector_unit = boom_vector / np.linalg.norm(
                        boom_vector)  # Could change to do it a single time and find it in a list
                    new_point_coordinates_body_fixed_frame = point + boom_vector_unit * current_wing_shifts[
                        j]  # Applying the panel shift
                    if (np.linalg.norm(new_point_coordinates_body_fixed_frame) > np.linalg.norm(boom_vector)):
                        raise Exception("Error. Wing shifted beyond the boom length")
                else:
                    new_point_coordinates_body_fixed_frame = point
            new_current_panel_coordinates[j, :] = new_point_coordinates_body_fixed_frame
        wing_coordinates_list.append(new_current_panel_coordinates)
    return wing_coordinates_list

@jit(nopython=True, cache=True)
def sliding_mass_dynamical_model(displacement_from_boom_origin_list,
                                 sliding_masses_list,
                                 sliding_mass_extreme_positions_list,
                                 sliding_mass_system_is_accross_two_booms,
                                 sliding_mass_unit_direction):
    # The displacement should be around the origin of the boom for an independent one, and wrt to the middle of the boom if it is an aligned one
    sliding_mass_system_CoM = np.array([0, 0, 0], dtype="float64")
    sliding_masses_positions_body_fixed_frame_list = []
    for i, current_mass in enumerate(sliding_masses_list):
        current_unit_direction = sliding_mass_unit_direction[i]
        current_displacement = displacement_from_boom_origin_list[i]
        current_extreme_positions = sliding_mass_extreme_positions_list[i]
        if ((not sliding_mass_system_is_accross_two_booms[i]) and current_displacement < 0):
            raise Exception("Error. Negative displacement for single-directional sliding mass.")

        if (sliding_mass_system_is_accross_two_booms[i]):
            current_sliding_mass_origin_body_fixed_frame = (current_extreme_positions[0, :] + current_extreme_positions[
                                                                                              1, :]) / 2
        else:
            current_sliding_mass_origin_body_fixed_frame = current_extreme_positions[0, :]

        current_mass_position_body_fixed_frame = current_sliding_mass_origin_body_fixed_frame + current_displacement * current_unit_direction

        # Check that it is still within bounds
        current_displacement_norm = np.linalg.norm(
            current_mass_position_body_fixed_frame - current_sliding_mass_origin_body_fixed_frame)
        if ((current_displacement_norm > max(
                np.linalg.norm(current_sliding_mass_origin_body_fixed_frame - current_extreme_positions[1, :]),
                np.linalg.norm(current_sliding_mass_origin_body_fixed_frame - current_extreme_positions[0,
                                                                              :])))):  # TODO: to correct because this is wrong
            raise Exception("Error. The requested displacement is larger than the sliding mass system capabilities.")
        sliding_mass_system_CoM += current_mass_position_body_fixed_frame * current_mass
        sliding_masses_positions_body_fixed_frame_list.append(current_mass_position_body_fixed_frame)

    return sliding_mass_system_CoM / sum(sliding_masses_list), sliding_masses_positions_body_fixed_frame_list