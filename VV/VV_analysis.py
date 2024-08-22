import numpy as np
from generalConstants import Project_directory
import matplotlib.pyplot as plt

opt_detumbling = False
one_vane_torque = False
tag = 'SRP_Torque'
state_history_array = np.loadtxt(Project_directory + f'/0_GeneratedData/VV_Data/state_history_VV_{tag}.dat')
dependent_variable_history_array = np.loadtxt(
    Project_directory + f'/0_GeneratedData/VV_Data/dependent_variable_history_VV_{tag}.dat')

t_secs = (state_history_array[:, 0] - state_history_array[0, 0])    # seconds
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
spacecraft_total_torque = dependent_variable_history_array[:, 20:23]

spacecraft_total_torque_norm = np.sqrt(spacecraft_total_torque[:, 0]**2 + spacecraft_total_torque[:, 1]**2 + spacecraft_total_torque[:, 2]**2)
spacecraft_srp_acceleration_norm = np.sqrt(spacecraft_srp_acceleration_vector[:, 0]**2 + spacecraft_srp_acceleration_vector[:, 1]**2 + spacecraft_srp_acceleration_vector[:, 2]**2)

sail_I = np.zeros((3, 3))
sail_I[0, 0] = 1  # kg m^2
sail_I[0, 1] = 0.5  # kg m^2
sail_I[0, 2] = -1  # kg m^2
sail_I[1, 0] = 0.5  # kg m^2
sail_I[1, 1] = 2  # kg m^2
sail_I[1, 2] = 1  # kg m^2
sail_I[2, 0] = -1  # kg m^2
sail_I[2, 1] = 1  # kg m^2
sail_I[2, 2] = 5  # kg m^2

angular_momentum_norm = []
omega_norm = []
sc_control_x, sc_control_y, sc_control_z = [], [], []
for i in range(len(omega_x)):
    omega = np.array([omega_x[i], omega_y[i], omega_z[i]])
    angular_momentum_norm.append(np.linalg.norm(np.dot(sail_I, omega)))

    T_gyro = -(np.cross(omega, np.dot(sail_I, omega)))
    T_control = spacecraft_total_torque[i, :] - T_gyro

    omega_norm.append(np.linalg.norm(omega))
    sc_control_x.append(T_control[0])
    sc_control_y.append(T_control[1])
    sc_control_z.append(T_control[2])

sc_control_x = np.array(sc_control_x)
sc_control_y = np.array(sc_control_y)
sc_control_z = np.array(sc_control_z)


fig = plt.figure()
subfigs = fig.subplots(2, 1)
subfigs[0].plot(t_secs[t_secs<20], sc_control_x[t_secs<20], label=r'$\tau_{x, \mathcal{B}}$', linestyle='-')
subfigs[0].plot(t_secs[t_secs<20], sc_control_y[t_secs<20], label=r'$\tau_{x, \mathcal{B}}$', linestyle='--')
subfigs[0].plot(t_secs[t_secs<20], sc_control_z[t_secs<20], label=r'$\tau_{x, \mathcal{B}}$', linestyle='-.')
subfigs[0].grid(True)
subfigs[0].legend()
subfigs[0].set_ylim(-0.2, 0.2)
subfigs[0].set_xlabel(r'$t$ [s]', fontsize=14)
subfigs[0].set_ylabel(r'$\vec{\tau}_{\mathcal{B}}$ [Nm]', fontsize=14)

subfigs[1].plot(t_secs[t_secs<20], omega_x[t_secs<20], label=r'$\omega_{x, \mathcal{B}}$', linestyle='-')
subfigs[1].plot(t_secs[t_secs<20], omega_y[t_secs<20], label=r'$\omega_{y, \mathcal{B}}$', linestyle='--')
subfigs[1].plot(t_secs[t_secs<20], omega_z[t_secs<20], label=r'$\omega_{z, \mathcal{B}}$', linestyle='-.')
subfigs[1].grid(True)
subfigs[1].legend()
subfigs[1].set_ylim(-2, 2)
subfigs[1].set_xlabel(r'$t$ [s]', fontsize=14)
subfigs[1].set_ylabel(r'$\vec{\omega}_{\mathcal{B}}$ [rad/s]', fontsize=14)
plt.subplots_adjust(hspace=0.7)

if (opt_detumbling):
    plt.savefig('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/VV/optimal_torque.png',
                dpi=600,
                bbox_inches='tight')

plt.figure()
plt.plot(t_secs, spacecraft_total_torque_norm)

plt.figure()
plt.plot(t_secs, spacecraft_srp_acceleration_norm)
plt.xlabel(r'$t$ [hours]', fontsize=14)
plt.ylabel(r'$||a_{SRP}||$ [m/s]', fontsize=14)
plt.grid(True)
if (one_vane_torque):
    plt.savefig('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/VV/SRP_acc_1_vane.png',
                dpi=600,
                bbox_inches='tight')


print(spacecraft_srp_acceleration_norm[0])
print(spacecraft_total_torque_norm[0])

plt.figure()
plt.plot(t_secs / (3600), np.rad2deg(omega_x))
plt.xlabel(r'$t$ [hours]', fontsize=14)
plt.ylabel(r'$\omega_{x, \mathcal{B}}$ [deg/s]', fontsize=14)
plt.grid(True)

if (one_vane_torque):
    plt.savefig('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/VV/omega_x_hist_1_vane.png',
                dpi=600,
                bbox_inches='tight')

plt.figure()
plt.plot(t_secs/(24*3600), keplerian_state[:, 0]/1000)

plt.figure()
plt.plot(t_secs/(24*3600), keplerian_state[:, 1])
plt.show()

