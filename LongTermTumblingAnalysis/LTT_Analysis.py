import numpy as np
import matplotlib.pyplot as plt
from longTermTumbling_ACS3Model import LTT_save_data_dir
from generalConstants import Project_directory
from pathlib import Path
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm

current_data_set = 'NoAsymmetry_ACS3'
analysis_data_dir = LTT_save_data_dir + '/LTT_NoAsymetry_data_ACS3'

states_history_dir = analysis_data_dir + "/states_history"
dependent_variable_history_dir = analysis_data_dir + "/dependent_variable_history"

# state history files
p = Path(states_history_dir)
state_history_files = [x for x in p.iterdir() if (not x.is_dir())]

# dependent variable history files
p = Path(dependent_variable_history_dir)
dependent_variable_history_files = [x for x in p.iterdir() if (not x.is_dir())]

LTT_results_dict = {}

# First handle the Keplerian orbit case for comparison
keplerian_dependent_variable_history_array = np.loadtxt('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/keplerian_orbit_dependent_variable_history.dat')
keplerian_state_history_array = np.loadtxt('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/keplerian_orbit_state_history.dat')
LTT_results_dict["keplerian" + "_time_array"] = (keplerian_state_history_array[:, 0] - keplerian_state_history_array[0, 0]) / (24 * 3600)
LTT_results_dict["keplerian" + "_kep_array"] = keplerian_dependent_variable_history_array[:, 1:7]
LTT_results_dict["keplerian" + "_apo_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 + keplerian_dependent_variable_history_array[:, 2])
LTT_results_dict["keplerian" + "_peri_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 - keplerian_dependent_variable_history_array[:, 2])
LTT_results_dict[f"keplerian" + "_omega_vector_array"] = np.array([0, 0, 0])

# Initialise some dictionary entries
LTT_results_dict["all" + "_time_array"] = []
LTT_results_dict["all" + "_kep_array"] = []
LTT_results_dict["all" + "_apo_array"] = []
LTT_results_dict["all" + "_peri_array"] = []
LTT_results_dict[f"all" + "_omega_vector_array"] = []
for current_state_history_path in state_history_files:
    current_state_history_array = np.loadtxt(current_state_history_path)

    # get the initial rotational velocity vector of the propagation
    l = str(current_state_history_path)[:-4].split('_')
    omega_z_rph = float(l[-1])
    omega_y_rph = float(l[-4])
    omega_x_rph = float(l[-7])
    omega_vector_rph = np.array([omega_x_rph, omega_y_rph, omega_z_rph])

    current_dependent_variable_history_path = dependent_variable_history_dir + f"/dependent_variable_history_omega_x_{omega_x_rph}_omega_y_{omega_y_rph}_omega_z_{omega_z_rph}.dat"
    current_dependent_variable_history_array = np.loadtxt(current_dependent_variable_history_path)

    print(omega_vector_rph)
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
        LTT_results_dict["default" + "_kep_array"] = current_kep_array
        LTT_results_dict["default" + "_apo_array"] = current_apo_array
        LTT_results_dict["default" + "_peri_array"] = current_peri_array

    else:
        for rot_vel_comp in [omega_vector_rph]:
            if ((f"{rot_vel_comp}" + "_time_array") not in LTT_results_dict):
                LTT_results_dict[f"{rot_vel_comp}" + "_time_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_kep_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_apo_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_peri_array"] = []
                LTT_results_dict[f"{rot_vel_comp}" + "_omega_vector_array"] = []
            LTT_results_dict[f"{rot_vel_comp}" + "_time_array"].append(current_time_array)
            LTT_results_dict[f"{rot_vel_comp}" + "_kep_array"].append(current_kep_array)
            LTT_results_dict[f"{rot_vel_comp}" + "_apo_array"].append(current_apo_array)
            LTT_results_dict[f"{rot_vel_comp}" + "_peri_array"].append(current_peri_array)
            LTT_results_dict[f"{rot_vel_comp}" + "_omega_vector_array"].append(omega_vector_rph)

        LTT_results_dict["all" + "_time_array"].append(current_time_array)
        LTT_results_dict["all" + "_kep_array"].append(current_kep_array)
        LTT_results_dict["all" + "_apo_array"].append(current_apo_array)
        LTT_results_dict["all" + "_peri_array"].append(current_peri_array)
        LTT_results_dict[f"all" + "_omega_vector_array"].append(omega_vector_rph)

labels_list = [r"Semi-major axis evolution, $a$, [km]",
               r"Eccentricity, $e$, [-]",
               r"Inclination, $i$, [deg]",
               r"Argument of periapsis, $\omega$, [deg]",
               r"Right Ascension of Ascending Node, $\Omega$, [deg]",
               r"True anomaly, $\theta$, [deg]"]

labels_change_list = [r"Semi-major axis change, $\Delta a$, [km]",
               r"Eccentricity change, $\Delta e$, [-]",
               r"Inclination change, $\Delta i$, [deg]",
               r"Argument of periapsis change, $\Delta \omega$, [deg]",
               r"Right Ascension of Ascending Node change, $\Delta \Omega$, [deg]",
               r"True anomaly change, $\Delta \theta$, [deg]"]

multiplying_factors = [1./1000, 1., 180./np.pi, 180./np.pi, 180./np.pi, 180./np.pi]
save_fig_labels = ['sma', 'ecc', "inc", "aop", "raan", "tranom"]

# omega magnitude colouring
color_index = []
for l in range(len(LTT_results_dict[f"all" + "_omega_vector_array"])):
    color_index.append(np.linalg.norm(LTT_results_dict[f"all" + "_omega_vector_array"][l]))
c_norm_omega_all = plt.Normalize(np.min(color_index), np.max(color_index))
cmap = plt.get_cmap('seismic')
c_omega_all = cmap(c_norm_omega_all(color_index))

for j in range(5):
    # Keplerian absolute variations
    plt.figure()
    for i in range(len(LTT_results_dict[f"all" + "_time_array"])):   #
        plt.plot(LTT_results_dict["all" + "_time_array"][i], LTT_results_dict["all" + "_kep_array"][i][:, j] * multiplying_factors[j],
                 c='royalblue',
                 label=f'{i}',
                 alpha=0.5,
                 linewidth=1)
    plt.grid(True)
    plt.xlabel(r"Time, $t$, [days]", fontsize=14)
    plt.ylabel(labels_list[j], fontsize=14)
    plt.savefig(Project_directory + f'/0_FinalPlots/LTT/' + current_data_set + '/' + save_fig_labels[j] + '.png',
                bbox_inches='tight',
                dpi=1200)

    # Keplerian change wrt to [0, 0, 0] and to Keplerian orbit
    for ref_key in ["default", "keplerian"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(LTT_results_dict[f"all" + "_time_array"])):   #
            t_arr = LTT_results_dict["all" + "_time_array"][i]

            current_spline = CubicSpline(LTT_results_dict[ref_key + "_time_array"],
                                                LTT_results_dict[ref_key + "_kep_array"][:, j])
            diff = LTT_results_dict["all" + "_kep_array"][i][:, j] - current_spline(LTT_results_dict["all" + "_time_array"][i])

            plt.plot(LTT_results_dict["all" + "_time_array"][i], diff * multiplying_factors[j],
                     c=c_omega_all[i],
                     label=f'{i}',
                     linewidth=1)

        plt.grid(True)
        plt.xlabel(r"Time, $t$, [days]", fontsize=14)
        plt.ylabel(labels_change_list[j], fontsize=14)
        fig.colorbar(cm.ScalarMappable(norm=c_norm_omega_all, cmap=cmap), ax=ax)
        plt.savefig(Project_directory + f'/0_FinalPlots/LTT/' + current_data_set + '/' + save_fig_labels[j] + f'_change_wrt_{ref_key}.png',
                    bbox_inches='tight',
                    dpi=1200)


ylabel_list_apo_peri = [r'Apocenter, $r_a$, [km]', r'Pericenter, $r_p$, [km]']
for k, feature_key in enumerate(["apo", "peri"]):
    # feature absolute variations
    plt.figure()
    for i in range(len(LTT_results_dict[f"all" + "_time_array"])):
        plt.plot(LTT_results_dict["all" + "_time_array"][i], LTT_results_dict["all" + f"_{feature_key}_array"][i]/1000,
                 c='royalblue',
                 label=f'{i}',
                 alpha=0.5,
                 linewidth=1)
    plt.grid(True)
    plt.xlabel(r"Time, $t$, [days]", fontsize=14)
    plt.ylabel(ylabel_list_apo_peri[k], fontsize=14)
    plt.savefig(Project_directory + f'/0_FinalPlots/LTT/' + current_data_set + '/' + f'{feature_key}.png',
                bbox_inches='tight',
                dpi=1200)

    # feature change wrt to [0, 0, 0] and to Keplerian orbit
    for ref_key in ["default", "keplerian"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(LTT_results_dict[f"all" + "_time_array"])):
            t_arr = LTT_results_dict["all" + "_time_array"][i]

            current_spline = CubicSpline(LTT_results_dict[ref_key + "_time_array"],
                                                LTT_results_dict[ref_key + f"_{feature_key}_array"])
            diff = LTT_results_dict["all" + f"_{feature_key}_array"][i] - current_spline(LTT_results_dict["all" + "_time_array"][i])

            plt.plot(LTT_results_dict["all" + "_time_array"][i], diff/1000,
                     c=c_omega_all[i],
                     label=f'{i}',
                     linewidth=1)
        plt.grid(True)
        plt.xlabel(r"Time, $t$, [days]", fontsize=14)
        plt.ylabel(ylabel_list_apo_peri[k], fontsize=14)
        fig.colorbar(cm.ScalarMappable(norm=c_norm_omega_all, cmap=cmap), ax=ax)
        plt.savefig(Project_directory + f'/0_FinalPlots/LTT/' + current_data_set + '/' + f'{feature_key}_change.png',
                    bbox_inches='tight',
                    dpi=1200)