import numpy as np
from generalConstants import Project_directory
import matplotlib.pyplot as plt
from constants import sail_I
colors_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
specific_folder_analysed = "/0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False"

omega_combinations = [(0.0, 0.0, -120.0),
                      (120.0, 0.0, 0.0),
                      (-85.0, -85.0, 0.0),
                      ]

combination_labels = [r'(0.0, 0.0, -12.0) deg/s',
                     r'(12.0, 0.0, 0.0) deg/s',
                     r'(-8.5, -8.5, -0.0) deg/s']

N = 100    # moving average number of terms
time_list_averaged = []
SRP_Torque_nom_list = []
total_torque_norm_list = []
detumbling_times_list = []
Gyro_torque_norm = []
for c in omega_combinations:
    current_state_history_array = np.loadtxt(Project_directory + specific_folder_analysed
                                     + f"/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat")
    current_dependent_variable_history_array = np.loadtxt(Project_directory + specific_folder_analysed
        + f"/dependent_variable_history/dependent_variable_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat")

    # Time array
    current_time_array_hours = (current_dependent_variable_history_array[:, 0] - current_dependent_variable_history_array[0, 0]) / 3600

    # SRP torque history
    current_Tx_array = current_dependent_variable_history_array[:, 11]
    current_Ty_array = current_dependent_variable_history_array[:, 12]
    current_Tz_array = current_dependent_variable_history_array[:, 13]
    current_SRP_Torque_norm = np.sqrt(current_Tx_array**2 + current_Ty_array**2 + current_Tz_array**2)

    current_total_torque_norm = current_dependent_variable_history_array[:, 20]

    # irradiance
    received_irradiance_shadow_function = current_dependent_variable_history_array[:, 7]

    # rotational velocity history
    omega_x_array_deg_s = np.rad2deg(current_state_history_array[:, 11])
    omega_y_array_deg_s = np.rad2deg(current_state_history_array[:, 12])
    omega_z_array_deg_s = np.rad2deg(current_state_history_array[:, 13])

    # gyroscopic torque
    omega_rad_s = current_state_history_array[:, 11:14]
    T_gyro = np.zeros((len(omega_rad_s[:, 0]), 3))
    T_gyro[:, 0] = -sail_I[1, 1] * omega_rad_s[:, 1]*omega_rad_s[:,2] + sail_I[2, 2]*omega_rad_s[:,2]*omega_rad_s[:,1]
    T_gyro[:, 1] = sail_I[0, 0] * omega_rad_s[:,0] * omega_rad_s[:,2] - sail_I[2, 2] * omega_rad_s[:,2] * omega_rad_s[:,0]
    T_gyro[:, 2] = -sail_I[0, 0] * omega_rad_s[:,0] * omega_rad_s[:,1] + sail_I[1, 1] * omega_rad_s[:,1] * omega_rad_s[:,0]

    print(T_gyro[:, 2])
    T_gyro_norm = np.sqrt(T_gyro[:, 0]**2 + T_gyro[:, 1]**2 + T_gyro[:, 2]**2)

    current_SRP_Torque_norm = np.convolve(current_SRP_Torque_norm, np.ones(N) / N, mode='valid')
    current_total_torque_norm = np.convolve(current_total_torque_norm, np.ones(N) / N, mode='valid')
    T_gyro_norm = np.convolve(T_gyro_norm, np.ones(N) / N, mode='valid')
    received_irradiance_shadow_function = np.convolve(received_irradiance_shadow_function, np.ones(N) / N, mode='valid')
    current_time_array_hours_averaged = np.convolve(current_time_array_hours, np.ones(N) / N, mode='valid')

    current_SRP_Torque_norm[received_irradiance_shadow_function < 0.1] = None
    current_total_torque_norm[received_irradiance_shadow_function < 0.1] = None
    T_gyro_norm[received_irradiance_shadow_function < 0.1] = None

    current_SRP_Torque_norm[current_SRP_Torque_norm < 1e-6] = None
    current_total_torque_norm[current_total_torque_norm < 1e-6] = None
    T_gyro_norm[T_gyro_norm < 1e-6] = None

    time_list_averaged.append(current_time_array_hours_averaged)
    SRP_Torque_nom_list.append(current_SRP_Torque_norm)
    total_torque_norm_list.append(current_total_torque_norm)
    Gyro_torque_norm.append(T_gyro_norm)

    # get detumbling time
    list_indices_zero_angles = np.where(np.sum(abs(current_dependent_variable_history_array[:, 21:29]), axis=1) == 0)[0]
    if (len(list_indices_zero_angles) != 0):
        current_detumbling_time_hours = (current_state_history_array[list_indices_zero_angles[0], 0]
                                         - current_state_history_array[0, 0]) / 3600
    else:
        current_detumbling_time_hours = None
    detumbling_times_list.append(current_detumbling_time_hours)

    fig, ax = plt.subplots()
    ax.plot(current_time_array_hours / 24, omega_x_array_deg_s, label=r'$\omega_{x, \mathcal{B}}$', color=colors_list[0])
    ax.plot(current_time_array_hours / 24, omega_y_array_deg_s, label=r'$\omega_{y, \mathcal{B}}$', color=colors_list[1])
    ax.plot(current_time_array_hours / 24, omega_z_array_deg_s, label=r'$\omega_{z, \mathcal{B}}$', color=colors_list[2])

    if (c == (0.0, 0.0, -120.0)):
        x1, x2, y1, y2 = 6.5, current_detumbling_time_hours/24, -0.05, 0.05  # subregion of the original image
        axins = ax.inset_axes(
            [0.005, 0.4, 0.3, 0.3],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        axins.plot(current_time_array_hours / 24, omega_x_array_deg_s, color=colors_list[0])
        axins.plot(current_time_array_hours / 24, omega_y_array_deg_s, color=colors_list[1])
        axins.plot(current_time_array_hours / 24, omega_z_array_deg_s, color=colors_list[2])

        ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.grid(True)
    plt.xlabel(r'$t$ [days]', fontsize=14)
    plt.ylabel(r'$\vec{\omega}_{\mathcal{B}}$ components [deg/s]', fontsize=14)
    plt.xlim((0, current_detumbling_time_hours/24))
    plt.legend()
    plt.savefig(
        '/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc'
        + f'/OmegaHist_{c}.png',
        dpi=600,
        bbox_inches='tight')

fig, ax = plt.subplots()
for i in range(len(omega_combinations)):
    ax.plot(time_list_averaged[i] / 24, SRP_Torque_nom_list[i], linestyle='-', color=colors_list[i])
    ax.plot(time_list_averaged[i] / 24, Gyro_torque_norm[i], linestyle=':', color=colors_list[i])
    ax.plot([], [], linestyle='-', linewidth=7, label=combination_labels[i], color=colors_list[i])

ax.plot([], [], linestyle='-', color='k', label='SRP Torque')
ax.plot([], [], linestyle=':', color='k', label='Gyro Torque')
ax.grid(True)
ax.legend(ncol=2, loc='upper left', prop={'size': 8})
ax.set_yscale('log')
ax.set_ylabel(r'$||\vec{T}||$ [Nm]', fontsize=14)
ax.set_xlabel(r'$t$ [days]', fontsize=14)
ax.set_xlim((0, max(detumbling_times_list)/24))
plt.savefig('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc'
            + '/TorqueMagnitudeComparison.png',
            dpi=600,
            bbox_inches='tight')
plt.show()



