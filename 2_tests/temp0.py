import numpy as np
import itertools
from MiscFunctions import chunks, divide_list
from LongTermTumblingAnalysis.longTermTumbling_ACS3Model import LTT_save_data_dir
import os

overwrite_previous_bool = False
n_processes = 15

# generate all combinations of rotational velocity components from a selected subset to analyse
# for later
# [-150, -135, -120, -105, -90, -75, -60, -50, -40, -30, -20, 150, 135, 120, 105, 90, 75, 60, 50, 40, 30, 20]
omega_x_list = np.arange(-15, 15 + 1, 5)
omega_y_list = np.arange(-15, 15 + 1, 5)
omega_z_list = np.arange(-15, 15 + 1, 5)

all_combinations = list(itertools.product(omega_x_list, omega_y_list, omega_z_list))
if (not overwrite_previous_bool):
    new_combs = []
    for comb in all_combinations:
        initial_rotational_velocity = np.array(
            [comb[0] * 2 * np.pi / 3600., comb[1] * 2 * np.pi / 3600, comb[2] * 2 * np.pi / 3600])
        rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
        tentative_file = LTT_save_data_dir + f'/LTT_NoAsymetry_data_ACS3/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
        if (os.path.isfile(tentative_file)):
            # if the file exists, skip this propagation
            continue
        else:
            new_combs.append(comb)
    all_combinations = new_combs
chunks_list = divide_list(all_combinations, n_processes)

print(len(chunks_list))
print(chunks_list)



