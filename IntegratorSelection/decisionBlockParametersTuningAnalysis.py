import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from tudatpy.util import compare_results, result2array
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix
import os
from pathlib import Path
from integratorSelectionSailModel import integrator_selection_data_directory
from generalConstants import Project_directory
from MiscFunctions import natural_keys

# subdirectory of the stored data
data_subdir = Project_directory + "/0_GeneratedData/IntegratorSelection_Data/decisionBlockTuning"

# load baseline trajectory - found through manual tuning to get a decent version
baseline_sub_folder = data_subdir + "/baseline_case/tol_0.0e+00"

## load ancillary information
with open(str(baseline_sub_folder) + "/ancillary_simulation_info.txt", 'r') as f:
    for l, line in enumerate(f):
        info = line.split("\t")[-1]
        if (l == 0):
            baseline_num_function_evaluations = int(info)
        elif (l == 5):
            baseline_computational_time = float(info)
        else:
            pass

## load state history and dependent variable array
baseline_state_history_array = np.loadtxt(baseline_sub_folder + f'/state_history.dat')
baseline_dependent_variable_array = np.loadtxt(baseline_sub_folder + f'/dependent_variable_history.dat')

baseline_state_history_dict, baseline_dependent_variable_dict = {}, {}
for j, time in enumerate(baseline_state_history_array[:, 0]):
    baseline_state_history_dict[time] = baseline_state_history_array[j, 1:]
    baseline_dependent_variable_dict[time] = baseline_dependent_variable_array[j, 1:]


## get detumbling time
list_indices_zero_angles = np.where(baseline_dependent_variable_array[:, 21] == 0)[0]
if (len(list_indices_zero_angles) != 0):
    baseline_detumbling_time_hours = (baseline_state_history_array[list_indices_zero_angles[0], 0]
                                      - baseline_state_history_array[0, 0]) / 3600
else:
    baseline_detumbling_time_hours = None

p = Path(data_subdir)
tuned_factors_sub_directories = sorted([x for x in p.iterdir() if x.is_dir()])
new_tolerance_sub_directories = []
for subdir in tuned_factors_sub_directories:
    folder_name = str(subdir).split('/')[-1]
    if (folder_name == "baseline_case"):
        continue
    new_tolerance_sub_directories.append(subdir)
tuned_factors_sub_directories = new_tolerance_sub_directories

# get tuned factors data
tuning_information = {}
for tuned_factor_dir in tuned_factors_sub_directories:
    current_factor_str = str(tuned_factor_dir).split('/')[-1]
    p = Path(tuned_factor_dir)
    tolerance_sub_directories = [x for x in p.iterdir() if x.is_dir()]

    # sort array
    temp_sort_array = np.empty((len(tolerance_sub_directories), 2), dtype=object)
    for t_id, t in enumerate(tolerance_sub_directories):
        temp_sort_array[t_id, 0] = float((str(t).split('/')[-1]).split('_')[-1])
        temp_sort_array[t_id, 1] = t
    temp_sort_array = temp_sort_array[temp_sort_array[:, 0].argsort()]
    tolerance_sub_directories = temp_sort_array[:, 1]

    time_array_list = []
    position_difference_norm_list, velocity_difference_norm_list, omega_difference_norm_list = [], [], []
    state_hist_array_list, dependent_variable_hist_array_list = [], []
    state_difference_array_list, dependent_variable_difference_array_list = [], []
    propagator_list, success_boolean_list = [], []
    detumbling_time_hours_list = []
    num_function_evaluations_list = []
    computational_time_list = []
    tolerance_list = []
    for tol_dir in tolerance_sub_directories:
        current_tol_str = str(tol_dir).split('/')[-1]
        print(f'---- {current_factor_str}= {current_tol_str.split("_")[-1]} ----')

        # extract states and dependent variables history
        current_tol_states_history_array = np.loadtxt(str(tol_dir) + '/state_history.dat')
        current_tol_dep_var_history_array = np.loadtxt(str(tol_dir) + '/dependent_variable_history.dat')

        # detumbling time determination
        current_list_indices_zero_angles = np.where(np.sum(current_tol_dep_var_history_array[:, 21:29], axis=1) == 0)[0]
        if (len(current_list_indices_zero_angles) != 0):
            detumbling_time_hours_list.append((current_tol_states_history_array[current_list_indices_zero_angles[0], 0]
                                               - current_tol_states_history_array[0, 0]) / 3600)
        else:
            detumbling_time_hours_list.append(None)

        # obtain ancillary data
        with open(str(tol_dir) + "/ancillary_simulation_info.txt", 'r') as f:
            for l, line in enumerate(f):
                info = line.split("\t")[-1]
                if (l == 0):
                    num_function_evaluations_list.append(int(info))
                elif (l == 4):
                    tolerance_list.append(float(info))
                elif (l == 5):
                    computational_time_list.append(float(info))
                else:
                    pass

        # some processing data
        # convert arrays to dictionaries to allow for easy comparison
        current_state_history_dict, current_dependent_variables_history_dict = {}, {}
        for j, time in enumerate(current_tol_states_history_array[:, 0]):
            current_state_history_dict[time] = current_tol_states_history_array[j, 1:]
            current_dependent_variables_history_dict[time] = current_tol_dep_var_history_array[j, 1:]

        # cut time nodes at which errors in interpolation due to Runge's phenomenon are expected
        current_state_history_time_array = current_tol_states_history_array[6:-6, 0]  # remove first six points and last six points due to interpolation error
        current_state_history_time_array = current_state_history_time_array[
            current_state_history_time_array > baseline_state_history_array[6, 0]]
        current_state_history_time_array = current_state_history_time_array[
            current_state_history_time_array < baseline_state_history_array[-6, 0]]

        # evaluate error with respect to benchmark
        state_history_difference_array = result2array(compare_results(baseline_state_history_dict, current_state_history_dict, current_state_history_time_array))
        dependent_variable_history_difference_array = result2array(compare_results(baseline_dependent_variable_dict, current_dependent_variables_history_dict, current_state_history_time_array))

        position_difference_norm = np.sqrt(np.sum(state_history_difference_array[:, 1:4] ** 2, axis=1))
        velocity_difference_norm = np.sqrt(np.sum(state_history_difference_array[:, 4:7] ** 2, axis=1))
        rotational_velocity_difference_norm = np.rad2deg(np.sqrt(np.sum(state_history_difference_array[:, 11:14] ** 2, axis=1)))

        time_array_list.append(current_state_history_time_array)
        state_hist_array_list.append(current_tol_states_history_array)
        dependent_variable_hist_array_list.append(current_tol_dep_var_history_array)

        state_difference_array_list.append(state_history_difference_array)
        dependent_variable_difference_array_list.append(dependent_variable_history_difference_array)

        position_difference_norm_list.append(position_difference_norm)
        velocity_difference_norm_list.append(velocity_difference_norm)
        omega_difference_norm_list.append(rotational_velocity_difference_norm)


    tuning_information[current_factor_str + '_detumbling_time_list'] = detumbling_time_hours_list
    tuning_information[current_factor_str + '_num_functions_evaluations_list'] = num_function_evaluations_list
    tuning_information[current_factor_str + '_computational_time_list'] = computational_time_list
    tuning_information[current_factor_str + '_tolerance_list'] = tolerance_list

    tuning_information[current_factor_str + '_time_array_list'] = time_array_list
    tuning_information[current_factor_str + '_state_history_array_list'] = state_hist_array_list
    tuning_information[current_factor_str + '_dependent_variable_array_list'] = dependent_variable_hist_array_list

    tuning_information[current_factor_str + '_state_history_difference_list'] = state_difference_array_list
    tuning_information[current_factor_str + '_dependent_variable_difference_list'] = dependent_variable_difference_array_list

    tuning_information[current_factor_str + '_position_difference_list'] = position_difference_norm_list
    tuning_information[current_factor_str + '_velocity_difference_list'] = velocity_difference_norm_list
    tuning_information[current_factor_str + '_omega_difference_list'] = omega_difference_norm_list

markers_list = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p",
                "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", "o", "v", "^", "<", ">"]

title_list = [
            r"Maximum $\Delta ||\vec{\omega}||$ tolerance, [deg/s]",
            r"Maximum $\Delta \hat{\omega}$ tolerance, [deg]",
            r"Maximum $\Delta \hat{n_{s}}$ tolerance, [deg]",
            r"Maximum $\Delta \hat{T_{v}}$ tolerance, [deg]",
            r"Maximum $\Delta ||\vec{T_{v}}||$ tolerance, [deg/s]",
            r"Maximum constraint violation in $\vec{T_{v}}$ allocation, [-]",
            r"$\vec{T_{v}}$ allocation optimisation iteration tolerance on objective, [Nm]",
            r"$\vec{T_{v}}$ allocation optimisation iteration tolerance on x, [Nm]",
            r"Vane angle determination tolerance, $\alpha_1$ and $\alpha_2$, [deg]",
            r"Start Golden section algorithm in vane angle determination, [deg]",
            r"$\vec{T_{v}}$ allocation optimisation objective weights"
               ]

for i, tuned_factor_dir in enumerate(tuned_factors_sub_directories):
    current_factor_str = str(tuned_factor_dir).split('/')[-1]

    # with computational time
    fig, ax = plt.subplots()
    ax.axvspan(baseline_computational_time*0.95, baseline_computational_time*1.05, alpha=0.5, color='red', label=r'Baseline $t_{comp} \pm 5\%$')
    ax.axhspan(baseline_detumbling_time_hours * 0.99, baseline_detumbling_time_hours * 1.01, alpha=0.5, color='blue',
               label=r'Baseline $\Delta t_{detumbling} \pm 1\%$')
    ax.set_title(title_list[i])
    for j, (x, y) in enumerate(zip(tuning_information[current_factor_str + '_computational_time_list'],
                    tuning_information[current_factor_str + '_detumbling_time_list'])):
        current_label = f"{float(tuning_information[current_factor_str + '_tolerance_list'][j]):.1e}"
        ax.scatter(tuning_information[current_factor_str + '_computational_time_list'][j],
                   tuning_information[current_factor_str + '_detumbling_time_list'][j], marker=markers_list[j],
                   label=current_label)
        #ax.annotate(f"{float(tuning_information[current_factor_str + '_tolerance_list'][j]):.1e}",
        #        xy=(x, y))
    ax.set_xlabel(r"Computational time, $t_{comp}$, [s]", fontsize=14)
    ax.set_ylabel(r"Detumbling time, $\Delta t_{detumbling}$, [hours]", fontsize=14)
    #plt.axvline(x=baseline_computational_time*0.95, color='r', linestyle='--', label=r'Baseline $t_{comp} \pm 5\%$')
    #plt.axvline(x=baseline_computational_time*1.05, color='r', linestyle='--')

    #plt.axhline(y=baseline_detumbling_time_hours, color='b', linestyle='--', label=r'Baseline $\Delta t_{detumbling}$')
    plt.grid(True)
    plt.legend()
    plt.savefig(Project_directory + f"/0_FinalPlots/DecisionBlockTuning/{current_factor_str}_computational_time.png",
                dpi=1200,
                bbox_inches='tight')
    #plt.close()

    # with number of simulation steps
    fig, ax = plt.subplots()
    ax.set_title(title_list[i])
    for j, (x, y) in enumerate(zip(tuning_information[current_factor_str + '_num_functions_evaluations_list'],
                    tuning_information[current_factor_str + '_detumbling_time_list'])):
        current_label = f"{float(tuning_information[current_factor_str + '_tolerance_list'][j]):.1e}"
        ax.scatter(tuning_information[current_factor_str + '_num_functions_evaluations_list'][j],
                   tuning_information[current_factor_str + '_detumbling_time_list'][j], marker=markers_list[j],
                   label=current_label)
        #ax.annotate(f"{float(tuning_information[current_factor_str + '_tolerance_list'][j]):.1e}",
        #        xy=(x, y))
    ax.set_xlabel(r"Number of function evaluations, $n$, [-]")
    ax.set_ylabel(r"Detumbling time, $\Delta t_{detumbling}$, [hours]")
    plt.axvline(x=baseline_num_function_evaluations, color='r', linestyle='--', label=r'Baseline $n$')
    plt.axhline(y=baseline_detumbling_time_hours, color='b', linestyle='--', label=r'Baseline $\Delta t_{detumbling}$')
    plt.grid(True)
    plt.legend()
    plt.savefig(Project_directory + f"/0_FinalPlots/DecisionBlockTuning/{current_factor_str}_int_func_eval.png",
                dpi=1200,
                bbox_inches='tight')
    plt.close()



