import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from MiscFunctions import compute_panel_geometrical_properties
from constants import *
boom_attachment_point = 0.64

VANES_BOOL = True
SHIFTED_PANELS_BOOL = False
keep_area = False

SLIDING_MASS_BOOL = False

# Boom points
boom1 = np.array([[0, 0, 0], [0, boom_length, 0]])
boom2 = np.array([[0, 0, 0], [boom_length, 0, 0]])
boom3 = np.array([[0, 0, 0], [0, -boom_length, 0]])
boom4 = np.array([[0, 0, 0], [-boom_length, 0, 0]])
boom_list = [boom1, boom2, boom3, boom4]

panel1 = np.array([[boom_attachment_point, 0, 0],
                   [boom_length, 0, 0],
                   [0, boom_length, 0],
                   [0, boom_attachment_point, 0]])

panel2 = np.array([[0, -boom_attachment_point, 0],
                    [0, -boom_length, 0],
                    [boom_length, 0, 0],
                    [boom_attachment_point, 0, 0]])

panel3 = np.array([[-boom_attachment_point, 0, 0],
                   [-boom_length, 0, 0],
                   [0, -boom_length, 0],
                   [0, -boom_attachment_point, 0]])

panel4 = np.array([[0, boom_attachment_point, 0],
                    [0, boom_length, 0],
                    [-boom_length, 0, 0],
                    [-boom_attachment_point, 0, 0]])

wings_coordinates_list = [panel1, panel2, panel3, panel4]
panels_optical_properties = [np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])] * 4

# vanes
vane_angle = np.deg2rad(30.)
vane_side_length = 0.5
vanes_rotation_matrices_list = [R.from_euler('z', 90., degrees=True).as_matrix(),
                                R.from_euler('z', 0., degrees=True).as_matrix(),
                                R.from_euler('z', 270., degrees=True).as_matrix(),
                                R.from_euler('z', 180., degrees=True).as_matrix(),
                                ]#R.from_euler('z', 45., degrees=True).as_matrix()

vanes_origin_list = [np.array([0., boom_length, 0.]),
                     np.array([boom_length, 0., 0.]),
                     np.array([0, -boom_length, 0.]),
                     np.array([-boom_length, 0., 0.]),
                     ]#np.array([np.cos(np.pi/4) * boom_length/np.sqrt(2), np.sin(np.pi/4) * boom_length/np.sqrt(2), 0.])

vane_coordinates_list = []
for i in range(len(vanes_origin_list)):
    current_vane_coords_body_frame_coords = vanes_origin_list[i]
    current_vane_rotation_matrix_body_to_vane = vanes_rotation_matrices_list[i]
    current_vane_rotation_matrix_vane_to_body = current_vane_rotation_matrix_body_to_vane

    second_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, -np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    third_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    current_vane_coords_body_frame_coords = np.vstack((current_vane_coords_body_frame_coords, second_point_body_frame, third_point_body_frame))
    vane_coordinates_list.append(current_vane_coords_body_frame_coords)

vanes_optical_properties = [np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])] * 4
#vanes_rotational_dof = [['x'], ['y'], ['x', 'y'], ['x', 'y']]

wings_rotation_matrices_list = [R.from_euler('z', -45, degrees=True).as_matrix(),
                                R.from_euler('z', -135, degrees=True).as_matrix(),
                                R.from_euler('z', -225, degrees=True).as_matrix(),
                                R.from_euler('z', -315, degrees=True).as_matrix()]

if (SHIFTED_PANELS_BOOL):
    if (keep_area == True):
        # Change panel coordinates to test shifted panels implementation, by ensuring that the panels can be shifted
        for i, wing_coords in enumerate(wings_coordinates_list):
            new_points = np.zeros(np.shape(wing_coords))
            for j, point in enumerate(wing_coords):
                point_wing_frame = np.matmul(np.linalg.inv(wings_rotation_matrices_list[i]), point)
                point_wing_frame += np.array([0, 0.5, 0])
                new_point = np.matmul(wings_rotation_matrices_list[i], point_wing_frame)
                new_points[j, :] = new_point
            wings_coordinates_list[i] = new_points

if (VANES_BOOL):
    acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
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
                                    vanes_optical_properties)

elif(SHIFTED_PANELS_BOOL):
    acs_object = sail_attitude_control_systems("shifted_wings", boom_list, sail_I, algorithm_constants)
    wing_area_list = []
    for i in range(len(wings_coordinates_list)):
        _, wing_area, _ = compute_panel_geometrical_properties(wings_coordinates_list[i])
        wing_area_list.append(wing_area)
    acs_object.set_shifted_panel_characteristics(wings_coordinates_list, wing_area_list, wings_rotation_matrices_list,
                                                 keep_area, 0, np.array([0, 0, 0]))
elif(SLIDING_MASS_BOOL):
    acs_object = sail_attitude_control_systems("sliding_masses", boom_list, sail_I, algorithm_constants)
    acs_object.set_sliding_masses_characteristics([10, 10], 0, np.array([0, 0, 0]), 1)

else:
    acs_object = sail_attitude_control_systems("None", boom_list, sail_I, algorithm_constants)

if (VANES_BOOL):
    sail = sail_craft("ACS3", 4, 4, wings_coordinates_list, vane_coordinates_list, panels_optical_properties, vanes_optical_properties,
                      sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, 0.00425, acs_object)
elif(SHIFTED_PANELS_BOOL):
    sail = sail_craft("ACS3", 4, 0, wings_coordinates_list, [], [], vanes_optical_properties,
                      sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, 0.00425, acs_object)
elif(SLIDING_MASS_BOOL):
    sail = sail_craft("ACS3", 4, 0, wings_coordinates_list, [], [], [],
                      sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, 0.00425, acs_object)
else:
    sail = sail_craft("ACS3", 4, 0, wings_coordinates_list, [], [], [],
                      sail_I, 16, 15.66, np.array([0., 0., 0.0]), 0.00425, 0.00425, acs_object)

wing_area_list = [sail.get_ith_panel_area(i, "Sail") for i in range(4)]
CoM = sail.get_sail_center_of_mass(0)
moving_masses_positions = sail.get_sail_moving_masses_positions(0)

wings_coordinates_list = [sail.get_ith_panel_coordinates(i, "Sail") for i in range(4)]
if VANES_BOOL: vane_coordinates_list = [sail.get_ith_panel_coordinates( i, "Vane") for i in range(4)]

# Plot booms
fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)
for boom in boom_list:
    ax.plot([boom[0][0], boom[1][0]], [boom[0][1], boom[1][1]],zs=[boom[0][2], boom[1][2]], color="k")

ax.add_collection3d(Poly3DCollection(wings_coordinates_list, alpha=0.5, zorder=0))
if VANES_BOOL: ax.add_collection3d(Poly3DCollection(vane_coordinates_list, alpha=0.5, zorder=0, color="g"))

vstack_centroid_surface_normal = np.array([0, 0, 0, 0, 0, 0])
if (VANES_BOOL): j_max = 2
else: j_max = 1
for j in range(j_max):
    if (j > 0):
        tp = "Vane"
        r = 4
    else:
        tp = "Sail"
        r = 4
    for i in range(r):
        panel_surface_normal_vector = sail.get_ith_panel_surface_normal(i, tp)
        panel_centroid = sail.get_ith_panel_centroid(i, tp)
        hstack_centroid_surface_normal = np.hstack((panel_centroid, panel_surface_normal_vector))
        vstack_centroid_surface_normal = np.vstack((vstack_centroid_surface_normal, hstack_centroid_surface_normal))

ax.quiver(vstack_centroid_surface_normal[:, 0],
        vstack_centroid_surface_normal[:, 1],
        vstack_centroid_surface_normal[:, 2],
        vstack_centroid_surface_normal[:, 3],
        vstack_centroid_surface_normal[:, 4],
        vstack_centroid_surface_normal[:, 5],
        color='r', arrow_length_ratio=0.1, zorder=20)

ax.set_box_aspect([1,1,1])
ax.set_xlim([-12, 12])  # set x-axis limits from 0 to 2
ax.set_ylim([-12, 12])  # set y-axis limits from 0 to 4
ax.set_zlim([-12, 12])  # set z-axis limits from 0 to 6
ax.set_proj_type('ortho')
ax.scatter(CoM[0], CoM[1], CoM[2], color="b", s=5)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
if SLIDING_MASS_BOOL:
    for key in moving_masses_positions.keys():
        masses_positions_list = moving_masses_positions[key]
        for mass_position in masses_positions_list:
            # TODO: maybe ensure that the mass is non-zero
            ax.scatter(mass_position[0], mass_position[1], mass_position[2], color="magenta", s=7)

plt.show()