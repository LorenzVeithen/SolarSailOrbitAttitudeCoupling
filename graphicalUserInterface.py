import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

import matplotlib.pyplot as plt
import numpy as np
from constants import *
from MiscFunctions import quiver_data_to_segments, set_axes_equal
from ACS_dynamical_models import vane_dynamical_model
import matplotlib.animation as animation
from MiscFunctions import compute_panel_geometrical_properties
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix
import matplotlib.gridspec as gridspec

plt.rcParams['animation.ffmpeg_path'] ='/opt/homebrew/bin/ffmpeg'
#plt.subplots_adjust(left=0.09, bottom=0.89, right=0.1, top=0.9, wspace=0.2, hspace=0.2 )    # Not really sure if this is doing anything

generate_mp4 = False

fps = 40
time = 25
quiver_length = 0.3 * R_E
quiver_widths = 1
quiver_length_attitude = 0.3 * boom_length
quiver_widths_attitude = 1
thr_previous_spacecraft_positions_fade_down = 1
thr_sun_rays = 1 * 24 * 3600

# Load data

state_history_array = np.loadtxt("PropagationData/DetumblingTorqueTest/state_history.dat")
dependent_variable_history_array = np.loadtxt(
    "PropagationData/DetumblingTorqueTest/dependent_variable_history.dat")

# Extract state history
t_hours = (state_history_array[:, 0] - state_history_array[0, 0]) / 3600    # hours
x_J2000 = state_history_array[:, 1]
y_J2000 = state_history_array[:, 2]
z_J2000 = state_history_array[:, 3]
vx_J2000 = state_history_array[:, 4]
vy_J2000 = state_history_array[:, 5]
vz_J2000 = state_history_array[:, 6]
quaternions_inertial_to_body_fixed_vector = state_history_array[:, 7:11]
omega_x = state_history_array[:, 11]
omega_y = state_history_array[:, 12]
omega_z = state_history_array[:, 13]

# Extract dependent variables
t_dependent_variables_hours = (dependent_variable_history_array[:, 0]-dependent_variable_history_array[0, 0])/3600
keplerian_state = dependent_variable_history_array[:, 1:7]
received_irradiance_shadow_function = dependent_variable_history_array[:, 7]
spacecraft_srp_acceleration_vector = dependent_variable_history_array[:, 8:11]
spacecraft_srp_torque_vector = dependent_variable_history_array[:, 11:14]
spacecraft_sun_relative_position = dependent_variable_history_array[:, 14:17]
earth_sun_relative_position = dependent_variable_history_array[:, 17:20]
spacecraft_total_torque_norm = dependent_variable_history_array[:, 20]
vanes_x_rotations = np.rad2deg(dependent_variable_history_array[:, 21:25])  # Note: this might need to be changed; is there a way to make this automatic?
vanes_y_rotations = np.rad2deg(dependent_variable_history_array[:, 25:29])  # Note: this might need to be changed; is there a way to make this automatic?

spacecraft_sun_relative_position_in_body_fixed_frame = np.zeros(np.shape(spacecraft_sun_relative_position))
for i in range(np.shape(t_dependent_variables_hours)[0]):
    current_R_BI = quaternion_entries_to_rotation_matrix(quaternions_inertial_to_body_fixed_vector[i, :].T)
    current_spacecraft_sun_relative_position = spacecraft_sun_relative_position[i, :]
    spacecraft_sun_relative_position_in_body_fixed_frame[i, :] = np.dot(current_R_BI, current_spacecraft_sun_relative_position)

spacecraft_srp_torque_norm = np.sqrt(spacecraft_srp_torque_vector[:, 0]**2 + spacecraft_srp_torque_vector[:, 1]**2 + spacecraft_srp_torque_vector[:, 2]**2)


gs = gridspec.GridSpec(4, 4)
fig1 = plt.figure()
fig1.tight_layout()
ax_orbit = fig1.add_subplot(gs[:2, :2], projection='3d')
ax_attitude = fig1.add_subplot(gs[:2, 2:], projection='3d')
ax_torque = fig1.add_subplot(gs[2, :])
ax_rotational_velocity = fig1.add_subplot(gs[3, :])

## Orbital side plot
# earth representation
u_E, v_E = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_E = R_E * np.cos(u_E)*np.sin(v_E)
y_E = R_E * np.sin(u_E)*np.sin(v_E)
z_E = R_E * np.cos(v_E)
ax_orbit.plot_wireframe(x_E, y_E, z_E, color="b", label="Earth")

# Real spacecraft orbit
previous_spacecraft_position = ax_orbit.plot([], [], [], c="b", alpha=0.3, label="Spacecraft position history")[0]
current_spacecraft_position = ax_orbit.scatter(x_J2000[0], y_J2000[0], z_J2000[0], c="k", label="Spacecraft")
ax_orbit.set_xlabel("X [m]")
ax_orbit.set_ylabel("Y [m]")
ax_orbit.set_zlabel("Z [m]")

# Sun rays
absolute_minimum = min([min(x_J2000), min(y_J2000), min(z_J2000)])
absolute_maximum = max([max(x_J2000), max(y_J2000), max(z_J2000)])
xgrid = np.linspace(absolute_minimum * 1.1, absolute_maximum * 1.1, 5)
ygrid = np.linspace(absolute_minimum * 1.1, absolute_maximum * 1.1, 5)
zgrid = np.linspace(absolute_minimum * 1.1, absolute_maximum * 1.1, 5)
Xg, Yg, Zg = np.meshgrid(xgrid, ygrid, zgrid)

# Define solar rays vector field
u = -np.ones_like(Xg) * earth_sun_relative_position[0, 0]  # Constant x-component
v = -np.ones_like(Yg) * earth_sun_relative_position[0, 1]  # Constant y-component
w = -np.ones_like(Zg) * earth_sun_relative_position[0, 2]  # Constant z-component

# Plot the sun rays and the sail normal vector
sun_rays = ax_orbit.quiver(Xg, Yg, Zg, u, v, w, normalize=True, color="gold", alpha=0.5,
                           linewidth=quiver_widths, length=quiver_length)
R_BI = quaternion_entries_to_rotation_matrix(quaternions_inertial_to_body_fixed_vector[0, :].T)
R_IB = np.linalg.inv(R_BI)
current_sail_normal = ax_orbit.quiver(x_J2000[0], y_J2000[0], z_J2000[0], R_IB[0, 2], R_IB[1, 2], R_IB[2, 2],
                                      color="k", normalize=True, linewidth=quiver_widths, length=quiver_length)
current_sail_srp_acceleration = ax_orbit.quiver(x_J2000[0], y_J2000[0], z_J2000[0],
                                                spacecraft_srp_acceleration_vector[0, 0],
                                                spacecraft_srp_acceleration_vector[0, 1],
                                                spacecraft_srp_acceleration_vector[0, 2],
                                                color="g", normalize=True, linewidth=quiver_widths,
                                                length=quiver_length, label="SRP acceleration")
set_axes_equal(ax_orbit)


## Attitude side plot
boom_plots_list = []
for boom in boom_list:
    boom_point_1_inertial = np.dot(R_IB, boom[0, :])
    boom_point_2_inertial = np.dot(R_IB, boom[1, :])
    boom_plots_list.append(ax_attitude.plot([boom_point_1_inertial[0], boom_point_2_inertial[0]],
                                            [boom_point_1_inertial[1], boom_point_2_inertial[1]],
                                            zs=[boom_point_1_inertial[2], boom_point_2_inertial[2]], color="k")[0])


wings_coordinates_in_inertial_frame = []
for wing in wings_coordinates_list:
    points_in_inertial_frame = np.zeros(np.shape(wing))
    for i, point in enumerate(wing):
        points_in_inertial_frame[i, :] = np.dot(R_IB, point)
    wings_coordinates_in_inertial_frame.append(points_in_inertial_frame)

collection_wings = Poly3DCollection(wings_coordinates_in_inertial_frame, alpha=0.5, zorder=0, facecolors='b', edgecolors='k')
ax_attitude.add_collection3d(collection_wings)

vstack_centroid_surface_normal_in_inertial_frame = np.array([0, 0, 0, 0, 0, 0])
for wing in wings_coordinates_in_inertial_frame:
    current_wing_centroid_inertial_frame, _, current_wing_surface_normal_in_inertial_frame = compute_panel_geometrical_properties(wing)
    hstack_centroid_surface_normal_in_inertial_frame = np.hstack((current_wing_centroid_inertial_frame, current_wing_surface_normal_in_inertial_frame))
    vstack_centroid_surface_normal_in_inertial_frame = np.vstack((vstack_centroid_surface_normal_in_inertial_frame, hstack_centroid_surface_normal_in_inertial_frame))

wings_normals_in_inertial_frame = ax_attitude.quiver(vstack_centroid_surface_normal_in_inertial_frame[:, 0],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 1],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 2],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 3],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 4],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 5],
                                                     color='r', arrow_length_ratio=0.1, zorder=20, length=1)

xgrid_attitude = np.linspace(-boom_length * 1.1, boom_length * 1.1, 5)
ygrid_attitude = np.linspace(-boom_length * 1.1, boom_length * 1.1, 5)
zgrid_attitude = np.linspace(-boom_length * 1.1, boom_length * 1.1, 5)
Xg_attitude, Yg_attitude, Zg_attitude = np.meshgrid(xgrid_attitude, ygrid_attitude, zgrid_attitude)

# Define solar rays vector field in the inertial frame axis moving with the orbital position
u_attitude = -np.ones_like(Xg_attitude) * spacecraft_sun_relative_position[0, 0]  # Constant x-component
v_attitude = -np.ones_like(Yg_attitude) * spacecraft_sun_relative_position[0, 1]  # Constant y-component
w_attitude = -np.ones_like(Zg_attitude) * spacecraft_sun_relative_position[0, 2]  # Constant z-component


# Plot the sun rays in the attitude plot
sun_rays_attitude = ax_attitude.quiver(Xg_attitude, Yg_attitude, Zg_attitude, u_attitude, v_attitude, w_attitude, normalize=True, color="gold", alpha=0.5,
                           linewidth=quiver_widths, length=quiver_length)

new_vane_coordinates_in_inertial_frame = []
for vane in vanes_coordinates_list:
    points_in_inertial_frame = np.zeros(np.shape(vane))
    for i, point in enumerate(vane):
        points_in_inertial_frame[i, :] = np.dot(R_IB, point)
    new_vane_coordinates_in_inertial_frame.append(points_in_inertial_frame)
collection_vanes = Poly3DCollection(new_vane_coordinates_in_inertial_frame, alpha=0.5, zorder=0, facecolors='g', edgecolors='k')
ax_attitude.add_collection3d(collection_vanes)

vstack_centroid_surface_normal_in_inertial_frame = np.array([0, 0, 0, 0, 0, 0])
for vane in new_vane_coordinates_in_inertial_frame:
    current_vane_centroid_in_inertial_frame, _, current_vane_surface_normal_in_inertial_frame = compute_panel_geometrical_properties(vane)
    hstack_centroid_surface_normal_in_inertial_frame = np.hstack((current_vane_centroid_in_inertial_frame, current_vane_surface_normal_in_inertial_frame))
    vstack_centroid_surface_normal_in_inertial_frame = np.vstack((vstack_centroid_surface_normal_in_inertial_frame, hstack_centroid_surface_normal_in_inertial_frame))

vanes_normals_in_inertial_frame = ax_attitude.quiver(vstack_centroid_surface_normal_in_inertial_frame[:, 0],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 1],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 2],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 3],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 4],
                                                     vstack_centroid_surface_normal_in_inertial_frame[:, 5],
                                                     color='r', arrow_length_ratio=0.1, zorder=20, length=1)
ax_attitude.set_xlabel("X [m]")
ax_attitude.set_ylabel("Y [m]")
ax_attitude.set_zlabel("Z [m]")
set_axes_equal(ax_attitude)

## SRP torque plot
srp_torque_plot = ax_torque.plot([t_hours[0]], [np.linalg.norm(spacecraft_srp_torque_vector[0, :])], label="SRP")[0]
total_torque_plot = ax_torque.plot([t_hours[0]], [np.linalg.norm(spacecraft_total_torque_norm[:])], label="Total")[0]
ax_torque.set_ylim(0, max(spacecraft_total_torque_norm) * 1.1)
ax_torque.set_xlabel('Time [hours]')
ax_torque.set_ylabel('Torque Magnitude [Nm]')
ax_torque.grid()

omega_x_plot = ax_rotational_velocity.plot([t_hours[0]], [omega_x[0]], label="omega_x")[0]
omega_y_plot = ax_rotational_velocity.plot([t_hours[0]], [omega_y[0]], label="omega_y")[0]
omega_z_plot = ax_rotational_velocity.plot([t_hours[0]], [omega_z[0]], label="omega_z")[0]
absolute_minimum = min([min(omega_x), min(omega_y), min(omega_z)])
absolute_maximum = max([max(omega_x), max(omega_y), max(omega_z)])
ax_rotational_velocity.set_ylim(absolute_minimum*1.1, absolute_maximum*1.1)
ax_rotational_velocity.set_xlabel('Time [hours]')
ax_rotational_velocity.set_ylabel('Rotational velocity [rad/s]')
ax_rotational_velocity.grid()
ax_rotational_velocity.legend()

def updateOrbit(frame):
    # for each frame, update the data stored on each artist.
    global sun_rays, current_sail_normal, current_sail_srp_acceleration, \
        sun_rays_attitude, vanes_normals_in_inertial_frame, wings_normals_in_inertial_frame

    # Update title
    fig1.suptitle(f"time={round(t_hours[frame], 2)} hours")

    # Orbit plot
    ## Inertial positions up to this point
    xd = x_J2000[:frame - 1]
    yd = y_J2000[:frame - 1]
    zd = z_J2000[:frame - 1]

    ## Update current position
    data_first = np.stack([[x_J2000[frame]], [y_J2000[frame]], [z_J2000[frame]]]).T
    current_spacecraft_position._offsets3d = (data_first[:, 0], data_first[:, 1], data_first[:, 2])

    ## Update current sail normal
    R_BI = np.linalg.inv(quaternion_entries_to_rotation_matrix(quaternions_inertial_to_body_fixed_vector[frame, :].T))
    R_IB = np.linalg.inv(R_BI)
    current_sail_normal.remove()
    current_sail_normal = ax_orbit.quiver(x_J2000[frame], y_J2000[frame], z_J2000[frame], R_IB[0, 2], R_IB[1, 2], R_IB[2, 2],
                                          color="k", normalize=True, linewidth=quiver_widths, length=quiver_length)

    ## Update current SRP acceleration
    current_sail_srp_acceleration.remove()
    current_sail_srp_acceleration = ax_orbit.quiver(x_J2000[frame], y_J2000[frame], z_J2000[frame],
                                                    spacecraft_srp_acceleration_vector[frame, 0],
                                                    spacecraft_srp_acceleration_vector[frame, 1],
                                                    spacecraft_srp_acceleration_vector[frame, 2],
                                                    color="g", normalize=True, linewidth=quiver_widths,
                                                    length=quiver_length)

    ## Update previous orbital points to leave a trail behind
    if frame > 1:
        data_previous = np.stack([xd, yd, zd]).T
        previous_spacecraft_position.set_xdata(data_previous[:, 0])
        previous_spacecraft_position.set_ydata(data_previous[:, 1])
        previous_spacecraft_position.set_3d_properties(data_previous[:, 2])

    ## Update sun rays in the orbit plot
    if (abs(t_hours[frame] % 1) < 1e-15):
        sun_rays.remove()
        u = -np.ones_like(Xg) * earth_sun_relative_position[frame, 0]  # Constant x-component
        v = -np.ones_like(Yg) * earth_sun_relative_position[frame, 1]  # Constant y-component
        w = -np.ones_like(Zg) * earth_sun_relative_position[frame, 2]  # Constant z-component
        sun_rays = ax_orbit.quiver(Xg, Yg, Zg, u, v, w, normalize=True, color="gold", alpha=0.5, linewidth=quiver_widths, length=quiver_length)

    sun_rays_attitude.remove()
    u_attitude = -np.ones_like(Xg_attitude) * spacecraft_sun_relative_position[frame, 0]  # Constant x-component
    v_attitude = -np.ones_like(Yg_attitude) * spacecraft_sun_relative_position[frame, 1]  # Constant y-component
    w_attitude = -np.ones_like(Zg_attitude) * spacecraft_sun_relative_position[frame, 2]  # Constant z-component
    sun_rays_attitude = ax_attitude.quiver(Xg_attitude, Yg_attitude, Zg_attitude, u_attitude, v_attitude, w_attitude,
                                           normalize=True, color="gold", alpha=0.5, linewidth=quiver_widths_attitude, length=quiver_length_attitude)


    # Torque plot
    srp_torque_plot.set_xdata(t_hours[:frame])
    srp_torque_plot.set_ydata(spacecraft_srp_torque_norm[:frame])

    total_torque_plot.set_xdata(t_hours[:frame])
    total_torque_plot.set_ydata(spacecraft_total_torque_norm[:frame])

    # Rotational velocity plot
    omega_x_plot.set_xdata(t_hours[:frame])
    omega_x_plot.set_ydata(omega_x[:frame])

    omega_y_plot.set_xdata(t_hours[:frame])
    omega_y_plot.set_ydata(omega_y[:frame])

    omega_z_plot.set_xdata(t_hours[:frame])
    omega_z_plot.set_ydata(omega_z[:frame])

    # Attitude plot
    new_vane_coordinates = vane_dynamical_model(rotation_x_deg=vanes_x_rotations[frame, :],
                         rotation_y_deg=vanes_y_rotations[frame, :],
                         number_of_vanes=len(vanes_coordinates_list),
                         vane_reference_frame_origin_list=vanes_origin_list,
                         vane_panels_coordinates_list=vanes_coordinates_list,
                         vane_reference_frame_rotation_matrix_list=vanes_rotation_matrices_list)

    new_vane_coordinates_in_inertial_frame = []
    for vane in new_vane_coordinates:
        points_in_inertial_frame = np.zeros(np.shape(vane))
        for i, point in enumerate(vane):
            points_in_inertial_frame[i, :] = np.dot(R_IB, point)
        new_vane_coordinates_in_inertial_frame.append(points_in_inertial_frame)
    collection_vanes.set_verts(new_vane_coordinates_in_inertial_frame)

    vstack_centroid_surface_normal_in_inertial_frame = np.array([0, 0, 0, 0, 0, 0])
    for vane in new_vane_coordinates_in_inertial_frame:
        current_vane_centroid_in_inertial_frame, _, current_vane_surface_normal_in_inertial_frame = compute_panel_geometrical_properties(
            vane)
        hstack_centroid_surface_normal_in_inertial_frame = np.hstack(
            (current_vane_centroid_in_inertial_frame, current_vane_surface_normal_in_inertial_frame))
        vstack_centroid_surface_normal_in_inertial_frame = np.vstack(
            (vstack_centroid_surface_normal_in_inertial_frame, hstack_centroid_surface_normal_in_inertial_frame))

    vanes_normals_in_inertial_frame.remove()
    vanes_normals_in_inertial_frame = ax_attitude.quiver(vstack_centroid_surface_normal_in_inertial_frame[:, 0],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 1],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 2],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 3],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 4],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 5],
                                                         color='r', arrow_length_ratio=0.1, zorder=20, length=1)

    wings_coordinates_in_inertial_frame = []
    for wing in wings_coordinates_list:
        points_in_inertial_frame = np.zeros(np.shape(wing))
        for i, point in enumerate(wing):
            points_in_inertial_frame[i, :] = np.dot(R_IB, point)
        wings_coordinates_in_inertial_frame.append(points_in_inertial_frame)
    collection_wings.set_verts(wings_coordinates_in_inertial_frame)

    vstack_centroid_surface_normal_in_inertial_frame = np.array([0, 0, 0, 0, 0, 0])
    for wing in wings_coordinates_in_inertial_frame:
        current_wing_centroid_inertial_frame, _, current_wing_surface_normal_in_inertial_frame = compute_panel_geometrical_properties(
            wing)
        hstack_centroid_surface_normal_in_inertial_frame = np.hstack(
            (current_wing_centroid_inertial_frame, current_wing_surface_normal_in_inertial_frame))
        vstack_centroid_surface_normal_in_inertial_frame = np.vstack(
            (vstack_centroid_surface_normal_in_inertial_frame, hstack_centroid_surface_normal_in_inertial_frame))

    wings_normals_in_inertial_frame.remove()
    wings_normals_in_inertial_frame = ax_attitude.quiver(vstack_centroid_surface_normal_in_inertial_frame[:, 0],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 1],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 2],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 3],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 4],
                                                         vstack_centroid_surface_normal_in_inertial_frame[:, 5],
                                                         color='r', arrow_length_ratio=0.1, zorder=20, length=1)

    for boom_plot in boom_plots_list:
        boom_point_1_inertial = np.dot(R_IB, boom[0, :])
        boom_point_2_inertial = np.dot(R_IB, boom[1, :])
        boom_plot.set_xdata([boom_point_1_inertial[0], boom_point_2_inertial[0]])
        boom_plot.set_ydata([boom_point_1_inertial[1], boom_point_2_inertial[1]])
        boom_plot.set_3d_properties([boom_point_1_inertial[2], boom_point_2_inertial[2]])

    # Change axis scales of the SRP and rotational velocity plots
    if frame > 1:
        ax_torque.set_xlim(0, t_hours[frame] + 0.2)
        ax_rotational_velocity.set_xlim(0, t_hours[frame] + 0.2)
    return (previous_spacecraft_position, current_sail_srp_acceleration, sun_rays, srp_torque_plot)

# Run animation
ani = animation.FuncAnimation(fig=fig1, func=updateOrbit, frames=len(t_hours), interval=5)
plt.show()
if generate_mp4:
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save('animation.mp4', writer=FFwriter)

