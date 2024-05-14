import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

import matplotlib.pyplot as plt
import numpy as np
from constants import *
from MiscFunctions import quiver_data_to_segments, set_axes_equal
import matplotlib.animation as animation
from MiscFunctions import compute_panel_geometrical_properties
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix

plt.rcParams['animation.ffmpeg_path'] ='/opt/homebrew/bin/ffmpeg'
#plt.subplots_adjust(left=0.09, bottom=0.89, right=0.1, top=0.9, wspace=0.2, hspace=0.2 )    # Not really sure if this is doing anything

generate_mp4 = False

fps = 40
time = 25
quiver_length = 0.3 * R_E
quiver_widths = 1
thr_previous_spacecraft_positions_fade_down = 1
thr_sun_rays = 1 * 24 * 3600

# Extract state history
state_history_array = np.loadtxt("PropagationData/state_history.dat")
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
dependent_variable_history_array = np.loadtxt("PropagationData/dependent_variable_history.dat")
t_dependent_variables_hours = (dependent_variable_history_array[:, 0]-dependent_variable_history_array[0, 0])/3600
keplerian_state = dependent_variable_history_array[:, 1:7]
received_irradiance_shadow_function = dependent_variable_history_array[:, 7]
spacecraft_srp_acceleration_vector = dependent_variable_history_array[:, 8:11]
spacecraft_srp_torque_vector = dependent_variable_history_array[:, 11:14]
sun_spacecraft_relative_position = dependent_variable_history_array[:, 14:17]
earth_sun_relative_position = dependent_variable_history_array[:, 17:20]

spacecraft_srp_torque_norm = np.sqrt(spacecraft_srp_torque_vector[:, 0]**2 + spacecraft_srp_torque_vector[:, 1]**2 + spacecraft_srp_torque_vector[:, 2]**2)

fig = plt.figure()
fig.tight_layout()
ax_orbit = fig.add_subplot(221, projection='3d')
ax_attitude = fig.add_subplot(222, projection='3d')
ax_srp_torque = fig.add_subplot(223)
ax_rotational_velocity = fig.add_subplot(224)

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
R_IB = quaternion_entries_to_rotation_matrix(quaternions_inertial_to_body_fixed_vector[0, :].T)
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
for boom in boom_list:
    ax_attitude.plot([boom[0][0], boom[1][0]], [boom[0][1], boom[1][1]],zs=[boom[0][2], boom[1][2]], color="k")

vstack_centroid_surface_normal = np.array([0, 0, 0, 0, 0, 0])
for wing in wings_coordinates_list:
    current_wing_centroid, _, current_wing_surface_normal = compute_panel_geometrical_properties(wing)
    hstack_centroid_surface_normal = np.hstack((current_wing_centroid, current_wing_surface_normal))
    vstack_centroid_surface_normal = np.vstack((vstack_centroid_surface_normal, hstack_centroid_surface_normal))
collection_wings = Poly3DCollection(wings_coordinates_list, alpha=0.5, zorder=0, facecolors='b', edgecolors='k')
ax_attitude.add_collection3d(collection_wings)
wings_normals = ax_attitude.quiver(vstack_centroid_surface_normal[:, 0],
        vstack_centroid_surface_normal[:, 1],
        vstack_centroid_surface_normal[:, 2],
        vstack_centroid_surface_normal[:, 3],
        vstack_centroid_surface_normal[:, 4],
        vstack_centroid_surface_normal[:, 5],
                                   color='r', arrow_length_ratio=0.1, zorder=20, length=1)

vstack_centroid_surface_normal = np.array([0, 0, 0, 0, 0, 0])
for vane in vanes_coordinates_list:
    current_vane_centroid, _, current_vane_surface_normal = compute_panel_geometrical_properties(vane)
    hstack_centroid_surface_normal = np.hstack((current_vane_centroid, current_vane_surface_normal))
    vstack_centroid_surface_normal = np.vstack((vstack_centroid_surface_normal, hstack_centroid_surface_normal))
collection_vanes = Poly3DCollection(vanes_coordinates_list, alpha=0.5, zorder=0, facecolors='g', edgecolors='k')
ax_attitude.add_collection3d(collection_vanes)
vanes_normals = ax_attitude.quiver(vstack_centroid_surface_normal[:, 0],
        vstack_centroid_surface_normal[:, 1],
        vstack_centroid_surface_normal[:, 2],
        vstack_centroid_surface_normal[:, 3],
        vstack_centroid_surface_normal[:, 4],
        vstack_centroid_surface_normal[:, 5],
        color='r', arrow_length_ratio=0.1, zorder=20, length=1)
set_axes_equal(ax_attitude)

## SRP torque plot
srp_torque_plot = ax_srp_torque.plot([t_hours[0]], [np.linalg.norm(spacecraft_srp_torque_vector[0, :])])[0]
ax_srp_torque.set_ylim(0, max(spacecraft_srp_torque_norm)*1.1)
ax_srp_torque.set_xlabel('Time [hours]')
ax_srp_torque.set_ylabel('SRP Torque [Nm]')
ax_srp_torque.grid()

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
    global sun_rays, current_sail_normal, current_sail_srp_acceleration
    xd = x_J2000[:frame - 1]
    yd = y_J2000[:frame - 1]
    zd = z_J2000[:frame - 1]
    # Update the first point
    data_first = np.stack([[x_J2000[frame]], [y_J2000[frame]], [z_J2000[frame]]]).T
    current_spacecraft_position._offsets3d = (data_first[:, 0], data_first[:, 1], data_first[:, 2])
    R_IB = quaternion_entries_to_rotation_matrix(quaternions_inertial_to_body_fixed_vector[frame, :].T)
    current_sail_normal.remove()
    current_sail_normal = ax_orbit.quiver(x_J2000[frame], y_J2000[frame], z_J2000[frame], R_IB[0, 2], R_IB[1, 2], R_IB[2, 2],
                                          color="k", normalize=True, linewidth=quiver_widths, length=quiver_length)
    current_sail_srp_acceleration.remove()
    current_sail_srp_acceleration = ax_orbit.quiver(x_J2000[frame], y_J2000[frame], z_J2000[frame],
                                                    spacecraft_srp_acceleration_vector[frame, 0],
                                                    spacecraft_srp_acceleration_vector[frame, 1],
                                                    spacecraft_srp_acceleration_vector[frame, 2],
                                                    color="g", normalize=True, linewidth=quiver_widths,
                                                    length=quiver_length)
    # Update previous points
    if frame > 1:
        data_previous = np.stack([xd, yd, zd]).T
        previous_spacecraft_position.set_xdata(data_previous[:, 0])
        previous_spacecraft_position.set_ydata(data_previous[:, 1])
        previous_spacecraft_position.set_3d_properties(data_previous[:, 2])

    ax_orbit.set_title(f"time={round(t_hours[frame], 2)} hours")
    if (frame % 10):    # TODO: change the update rate to much less often
        sun_rays.remove()
        # Define constant vector components for the field
        u = -np.ones_like(Xg) * earth_sun_relative_position[frame, 0]  # Constant x-component
        v = -np.ones_like(Yg) * earth_sun_relative_position[frame, 1]  # Constant y-component
        w = -np.ones_like(Zg) * earth_sun_relative_position[frame, 2]  # Constant z-component
        sun_rays = ax_orbit.quiver(Xg, Yg, Zg, u, v, w, normalize=True, color="gold", alpha=0.5, linewidth=quiver_widths, length=quiver_length)


    srp_torque_plot.set_xdata(t_hours[:frame])
    srp_torque_plot.set_ydata(spacecraft_srp_torque_norm[:frame])

    omega_x_plot.set_xdata(t_hours[:frame])
    omega_x_plot.set_ydata(omega_x[:frame])

    omega_y_plot.set_xdata(t_hours[:frame])
    omega_y_plot.set_ydata(omega_y[:frame])

    omega_z_plot.set_xdata(t_hours[:frame])
    omega_z_plot.set_ydata(omega_z[:frame])
    if frame > 1:
        ax_srp_torque.set_xlim(0, t_hours[frame]+0.2)
        ax_rotational_velocity.set_xlim(0, t_hours[frame] + 0.2)
    return (previous_spacecraft_position, current_sail_srp_acceleration, sun_rays, srp_torque_plot)

ani = animation.FuncAnimation(fig=fig, func=updateOrbit, frames=len(t_hours), interval=50)
plt.show()
if generate_mp4:
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save('animation.mp4', writer=FFwriter)

"""
plt.figure()
plt.plot(t_hours, received_irradiance_shadow_function, label='Irradiance')
plt.plot(t_hours, spacecraft_srp_torque_norm/max(spacecraft_srp_torque_norm), label='SRP torque')
"""

