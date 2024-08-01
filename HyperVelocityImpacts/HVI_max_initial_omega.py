import numpy as np
from generalConstants import Project_directory
import matplotlib.pyplot as plt
from pathlib import Path


analysis_data_dir = Project_directory + f'/0_GeneratedData/DetumblingAnalysis/SingleOrbit/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_True'
states_history_dir = analysis_data_dir + "/states_history"

# state history files
p = Path(states_history_dir)
state_history_files = [x for x in p.iterdir() if (not x.is_dir())]


initial_omega_list_two_axis = []
initial_omega_list_three_axis = []
final_omega_list_two_axis = []
final_omega_list_three_axis = []
for current_state_history_path in state_history_files:
    if (str(current_state_history_path).split('/')[-1][0] == '.'):
        continue
    current_state_history_array = np.loadtxt(current_state_history_path)

    if (len(current_state_history_array[:, 0]) < 5):
        # some propagations may be broken, just remove them here
        print("skip")
        continue
    # get the initial rotational velocity vector of the propagation
    l = str(current_state_history_path)[:-4].split('_')
    omega_z_rph = float(l[-1])
    omega_y_rph = float(l[-4])
    omega_x_rph = float(l[-7])

    if (omega_x_rph == omega_y_rph == 0 or omega_z_rph == 0):
        continue

    initial_omega_vector_deg_s = np.array([omega_x_rph, omega_y_rph, omega_z_rph]) / 10.
    initial_omega_vector_rph = np.array([omega_x_rph, omega_y_rph, omega_z_rph])

    print(tuple(initial_omega_vector_rph))

    # extract Keplerian elements history
    # 1: Semi-major Axis. 2: Eccentricity. 3: Inclination. 4: Argument of Periapsis.
    # 5. Right Ascension of the Ascending Node. 6: True Anomaly.
    current_time_array = (current_state_history_array[:, 0]-current_state_history_array[0, 0])/(60)

    # rotational velocity history
    omega_x_array_deg_s = np.rad2deg(current_state_history_array[:, 11])
    omega_y_array_deg_s = np.rad2deg(current_state_history_array[:, 12])
    omega_z_array_deg_s = np.rad2deg(current_state_history_array[:, 13])

    omega_norm_deg_s = np.sqrt(omega_x_array_deg_s**2 + omega_y_array_deg_s**2 + omega_z_array_deg_s**2)

    if (omega_z_rph == 0):
        final_omega_list_two_axis.append(omega_norm_deg_s[-1])
        initial_omega_list_two_axis.append(np.linalg.norm(initial_omega_vector_deg_s))
    else:
        final_omega_list_three_axis.append(omega_norm_deg_s[-1])
        initial_omega_list_three_axis.append(np.linalg.norm(initial_omega_vector_deg_s))

final_omega_list_two_axis = np.array(final_omega_list_two_axis)
initial_omega_list_two_axis = np.array(initial_omega_list_two_axis)
final_omega_list_three_axis = np.array(final_omega_list_three_axis)
initial_omega_list_three_axis = np.array(initial_omega_list_three_axis)

relative_change_two_axis = (final_omega_list_two_axis - initial_omega_list_two_axis)   # * 100/initial_omega_list_two_axis  # %
relative_change_three_axis = (final_omega_list_three_axis - initial_omega_list_three_axis)   # * 100/initial_omega_list_three_axis  # %

plt.figure()
#plt.scatter(initial_omega_list_two_axis, relative_change_two_axis, label='2-axis')
plt.scatter(initial_omega_list_three_axis, relative_change_three_axis, label='3-axis')
plt.grid(True)
plt.legend()
plt.xlabel(r'$||\omega_{0}|| [deg/s]$', fontsize=14)
plt.ylabel(r'$\Delta ||\omega||$ [deg/s]', fontsize=14)
plt.show()