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

# subdirectory of the stored data
data_subdir = Project_directory + "/0_GeneratedData/IntegratorSelection_Data/decisionBlockTuning"

# load baseline trajectory - found through manual tuning to get a decent version
baseline_sub_folder = data_subdir + "/baseline_case/tol_0"

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

## get detumbling time
list_indices_zero_angles = np.where(baseline_dependent_variable_array[:, 21] == 0)[0]
if (len(list_indices_zero_angles) != 0):
    baseline_detumbling_time_hours = (baseline_state_history_array[list_indices_zero_angles[0], 0]
                                      - baseline_state_history_array[0, 0]) / 3600
else:
    baseline_detumbling_time_hours = None

p = Path(data_subdir)
tuned_factors_sub_directories = [x for x in p.iterdir() if x.is_dir()]
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

    for tol_dir in tolerance_sub_directories:
        current_tol_str = str(tuned_factor_dir).split('/')[-1]
        print(f'---- {current_factor_str}= {current_tol_str.split("_")[-1]} ----')

        # extract states and dependent variables history
        current_tol_states_history = np.loadtxt(str(tol_dir) + '/state_history.dat')
        current_tol_dep_var_history = np.loadtxt(str(tol_dir) + '/dependent_variable_history.dat')


