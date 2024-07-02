import numpy as np
import matplotlib.pyplot as plt
from longTermTumbling_ACS3Model import analysis_save_data_dir
from generalConstants import Project_directory
from pathlib import Path
from plottingRoutines import absolute_evolution_plot_LTT, relative_evolution_plot_LTT

current_data_set = 'NoAsymmetry_ACS3'
analysis_data_dir = analysis_save_data_dir + f'/LTT_NoAsymetry_data_ACS3/LEO_ecc_0.0/'
thinning_factor = 10

keplerian_keys = ['sma', 'ecc', "inc", "aop", "raan", "tranom"]
omega_components_id = ['x', 'y', 'z']

# prepare labels and scaling factors
labels_list = [r"Semi-major axis evolution, $a$, [km]",
               r"Eccentricity, $e$, [-]",
               r"Inclination, $i$, [deg]",
               r"Argument of periapsis, $\omega$, [deg]",
               r"RAAN, $\Omega$, [deg]",
               r"True anomaly, $\theta$, [deg]",
               r'Apocenter, $r_a$, [km]',
               r'Pericenter, $r_p$, [km]']

labels_change_list = [r"Semi-major axis change, $\Delta a$, [km]",
                r"Eccentricity change, $\Delta e$, [-]",
                r"Inclination change, $\Delta i$, [deg]",
                "Argument of periapsis change, \n" + r"$\Delta \omega$, [deg]",
                "RAAN change, " + r"$\Delta \Omega$, [deg]",
                r"True anomaly change, $\Delta \theta$, [deg]",
                r'Apocenter change, $\Delta r_a$, [km]',
                r'Pericenter change, $\Delta r_p$, [km]']

colour_bar_label = [r'Rotational velocity vector norm, $||\vec{\omega}||$, [deg/s]',
                    r'Body frame x-rotational velocity, $\omega_{x}$, [deg/s]',
                    r'Body frame y-rotational velocity, $\omega_{y}$, [deg/s]',
                    r'Body frame z-rotational velocity, $\omega_{z}$, [deg/s]',
                    r'Sail-plane rotational velocity norm, $||\vec{\omega_{xy}}||$, [deg/s]']
save_fig_cb_label = ["magnitude", "x", "y", "z", "xy_magnitude"]

multiplying_factors = [1./1000, 1., 180./np.pi, 180./np.pi, 180./np.pi, 180./np.pi, 1./1000, 1./1000]
variables_list = ['sma', 'ecc', "inc", "aop", "raan", "tranom", "apo", "peri"]


states_history_dir = analysis_data_dir + "/states_history"
dependent_variable_history_dir = analysis_data_dir + "/dependent_variable_history"
# state history files
p = Path(states_history_dir)
state_history_files = [x for x in p.iterdir() if (not x.is_dir())]

# add the path with zero rotational velocity in case it was not in the original bundle
omega_is_zero_vector_data = Path('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/LTT_NoAsymetry_data_ACS3/LEO_ecc_0.0/states_history/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0.dat')
if (((omega_is_zero_vector_data in state_history_files) == False) and (omega_is_zero_vector_data.exists())):
    state_history_files += [omega_is_zero_vector_data]


LTT_results_dict = {}

# First handle the Keplerian orbit case for comparison
keplerian_dependent_variable_history_array = np.loadtxt('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/LTT_NoAsymetry_data_ACS3/LEO_ecc_0.0/keplerian_orbit_dependent_variable_history.dat')
keplerian_state_history_array = np.loadtxt('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/LTT_NoAsymetry_data_ACS3/LEO_ecc_0.0/keplerian_orbit_state_history.dat')
LTT_results_dict["keplerian" + "_time_array"] = (keplerian_state_history_array[:, 0] - keplerian_state_history_array[0, 0]) / (24 * 3600)
for i in range(6):
    LTT_results_dict["keplerian" + f"_{keplerian_keys[i]}_array"] = keplerian_dependent_variable_history_array[:, i+1]
LTT_results_dict["keplerian" + "_apo_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 + keplerian_dependent_variable_history_array[:, 2])
LTT_results_dict["keplerian" + "_peri_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 - keplerian_dependent_variable_history_array[:, 2])
LTT_results_dict[f"keplerian" + "_omega_vector_array"] = np.array([0, 0, 0])

# Initialise some dictionary entries
LTT_results_dict["all" + "_time_array"] = []
for i in range(6):
    LTT_results_dict["all" + f"_{keplerian_keys[i]}_array"] = []
LTT_results_dict["all" + "_apo_array"] = []
LTT_results_dict["all" + "_peri_array"] = []
LTT_results_dict[f"all" + "_omega_vector_array"] = []


for current_state_history_path in state_history_files:
    if (str(current_state_history_path).split('/')[-1][0] == '.'):
        continue
    print(str(current_state_history_path).split('/')[-1])
    current_state_history_array = np.loadtxt(current_state_history_path)
    current_state_history_array = current_state_history_array[::thinning_factor]
    # get the initial rotational velocity vector of the propagation
    l = str(current_state_history_path)[:-4].split('_')
    omega_z_rph = float(l[-1])
    omega_y_rph = float(l[-4])
    omega_x_rph = float(l[-7])

    current_dependent_variable_history_path = dependent_variable_history_dir + f"/dependent_variable_history_omega_x_{omega_x_rph}_omega_y_{omega_y_rph}_omega_z_{omega_z_rph}.dat"
    current_dependent_variable_history_array = np.loadtxt(current_dependent_variable_history_path)
    current_dependent_variable_history_array = current_dependent_variable_history_array[::thinning_factor]
    omega_vector_deg_per_sec = np.array([omega_x_rph, omega_y_rph, omega_z_rph])/10
    omega_vector_rph = np.array([omega_x_rph, omega_y_rph, omega_z_rph])
    #print(omega_vector_rph)    # print the rotations per hour

    # extract Keplerian elements history
    # 1: Semi-major Axis. 2: Eccentricity. 3: Inclination. 4: Argument of Periapsis.
    # 5. Right Ascension of the Ascending Node. 6: True Anomaly.
    current_time_array = (current_dependent_variable_history_array[:, 0]-current_dependent_variable_history_array[0, 0])/(24*3600)
    current_kep_array = current_dependent_variable_history_array[:, 1:7]

    # apogee and perigee
    current_apo_array = current_kep_array[:, 0] * (1 + current_kep_array[:, 1])
    current_peri_array = current_kep_array[:, 0] * (1 - current_kep_array[:, 1])


    if (omega_x_rph==0 and omega_y_rph==0 and omega_z_rph==0):
        LTT_results_dict["default" + "_time_array"] = current_time_array
        for i in range(6):
            LTT_results_dict["default" + f"_{keplerian_keys[i]}_array"] = current_kep_array[:, i]
        LTT_results_dict["default" + "_apo_array"] = current_apo_array
        LTT_results_dict["default" + "_peri_array"] = current_peri_array

    else:
        for id, rot_vel_comp in enumerate(omega_vector_rph):
            if ((f"{rot_vel_comp}" + "_time_array") not in LTT_results_dict):
                LTT_results_dict[f"{rot_vel_comp}" + "_time_array"] = []
                for i in range(6):
                    LTT_results_dict[f"{rot_vel_comp}" + f"_{keplerian_keys[i]}_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_apo_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_peri_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_omega_vector_array"] = []
            LTT_results_dict[f"{rot_vel_comp}" + "_time_array"].append(current_time_array)
            for i in range(6):
                LTT_results_dict[f"{rot_vel_comp}" + f"_{keplerian_keys[i]}_array"].append(current_kep_array[:, i])
            LTT_results_dict[f"{rot_vel_comp}" + "_apo_array"].append(current_apo_array)
            LTT_results_dict[f"{rot_vel_comp}" + "_peri_array"].append(current_peri_array)
            LTT_results_dict[f"{rot_vel_comp}" + "_omega_vector_array"].append(omega_vector_deg_per_sec)

        LTT_results_dict["all" + "_time_array"].append(current_time_array)
        for i in range(6):
            LTT_results_dict["all" + f"_{keplerian_keys[i]}_array"].append(current_kep_array[:, i])
        LTT_results_dict["all" + "_apo_array"].append(current_apo_array)
        LTT_results_dict["all" + "_peri_array"].append(current_peri_array)
        LTT_results_dict[f"all" + "_omega_vector_array"].append(omega_vector_deg_per_sec)

processed_array_dict = LTT_results_dict
# scale variables as desired
for i in range(len(variables_list)):
    processed_array_dict["default" + f"_{variables_list[i]}_array"] = (
            np.array(processed_array_dict["default" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
    processed_array_dict["keplerian" + f"_{variables_list[i]}_array"] = (
            np.array(processed_array_dict["keplerian" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
    for j in range(len(processed_array_dict["all" + f"_{variables_list[i]}_array"])):
        processed_array_dict["all" + f"_{variables_list[i]}_array"][j] = (
                processed_array_dict["all" + f"_{variables_list[i]}_array"][j] * multiplying_factors[i])

# omega lists for colour bars
all_omega_norm_list = []
all_omega_x_list, all_omega_y_list, all_omega_z_list = [], [], []
all_omega_xy_norm_list = []
for l in range(len(LTT_results_dict[f"all" + "_omega_vector_array"])):
    all_omega_norm_list.append(np.linalg.norm(LTT_results_dict[f"all" + "_omega_vector_array"][l]))
    all_omega_x_list.append(LTT_results_dict[f"all" + "_omega_vector_array"][l][0])
    all_omega_y_list.append(LTT_results_dict[f"all" + "_omega_vector_array"][l][1])
    all_omega_z_list.append(LTT_results_dict[f"all" + "_omega_vector_array"][l][2])
    all_omega_xy_norm_list.append(np.sqrt(LTT_results_dict[f"all" + "_omega_vector_array"][l][0]**2
                                          + LTT_results_dict[f"all" + "_omega_vector_array"][l][1]**2))

time_label = r"Time, $t$, [days]"
for i in range(8):
    # Keplerian absolute variations
    save_path = Project_directory + f'/0_FinalPlots/LTT/' + current_data_set + '/' + variables_list[i] + '.png'
    absolute_evolution_plot_LTT(LTT_results_dict[f"all" + "_time_array"],
                                processed_array_dict["all" + f"_{variables_list[i]}_array"],
                                time_label,
                                labels_list[i],
                                save_path,
                                close_fig_bool=True)

    # Keplerian change wrt to [0, 0, 0] and to Keplerian orbit with omega magnitude
    for cb_id, cb_value in enumerate([all_omega_norm_list, all_omega_x_list, all_omega_y_list, all_omega_z_list, all_omega_xy_norm_list]):
        for ref_key in ["default", "keplerian"]:
            if (ref_key == "default" and omega_is_zero_vector_data.exists()==False):
                continue

            save_path = Project_directory + f'/0_FinalPlots/LTT/' + current_data_set + '/' + variables_list[
                i] + f'_change_wrt_{ref_key}_cbar_omega_{save_fig_cb_label[cb_id]}.png'
            relative_evolution_plot_LTT(LTT_results_dict[f"all" + "_time_array"],
                                        processed_array_dict["all" + f"_{variables_list[i]}_array"],
                                        time_label,
                                        labels_change_list[i],
                                        LTT_results_dict[ref_key + "_time_array"],
                                        processed_array_dict[ref_key + f"_{variables_list[i]}_array"],
                                        cb_value,
                                        colour_bar_label[cb_id],
                                        save_path,
                                        close_fig_bool=True)


