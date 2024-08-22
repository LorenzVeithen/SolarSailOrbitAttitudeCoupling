from solarSailPropagationFunction import runPropagationAnalysis
from mpi4py import MPI
import numpy as np
import itertools
import sys
import random

random.seed(42)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

# argv[0] is the executable
optical_model_mode = int(sys.argv[1])  # 0: ACS3 optical model, 1: double-sided  ideal model, 2: single-sided ideal model
sma_ecc_inc_combination_mode = int(sys.argv[2])    # see below for the combinations
include_shadow_b = int(sys.argv[3])     #0: False (no shadow), 1: True (with shadow)

optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][optical_model_mode]

# single axis
omega_list_single = list(np.arange(-150, 150 + 1, 10))
omega_list_single.remove(0)
all_single_axis_combinations = (list(itertools.product(omega_list_single, [0], [0])) + list(itertools.product([0], omega_list_single, [0]))
                                + list(itertools.product([0], [0], omega_list_single)))

# double axis
omega_list_double = [100, 85, 70, 55, 40, 30, 20, 10]   # -100, -85, -70, -55, -40, -30, -20, -10,
all_double_axis_combinations = []
for omega in omega_list_double:
    all_double_axis_combinations.append((omega, omega, 0))
    all_double_axis_combinations.append((0, omega, omega))
    all_double_axis_combinations.append((omega, 0, omega))
    all_double_axis_combinations.append((omega, -omega, 0))
    all_double_axis_combinations.append((0, omega, -omega))
    all_double_axis_combinations.append((omega, 0, -omega))
    all_double_axis_combinations.append((-omega, omega, 0))
    all_double_axis_combinations.append((0, -omega, omega))
    all_double_axis_combinations.append((-omega, 0, omega))
    all_double_axis_combinations.append((-omega, -omega, 0))
    all_double_axis_combinations.append((0, -omega, -omega))
    all_double_axis_combinations.append((-omega, 0, -omega))

# triple axis
omega_list_triple = [-85, -70, -55, -40, -30, -20, -10, 85, 70, 55, 40, 30, 20, 10]

all_triple_axis_combinations = list(itertools.product(omega_list_triple, omega_list_triple, omega_list_triple))
sampled_triple_axis_combinations = random.sample(all_triple_axis_combinations, 250)
for omega in omega_list_triple:
    if ((omega, omega, omega) not in sampled_triple_axis_combinations):
        sampled_triple_axis_combinations.append((omega, omega, omega))

#all_combinations = all_single_axis_combinations + all_double_axis_combinations + sampled_triple_axis_combinations
all_combinations = all_single_axis_combinations + all_double_axis_combinations
print(len(all_combinations))

print(f"hello from rank {rank}")
if (rank==0):
    runPropagationAnalysis(all_combinations,
                           optical_mode_str,
                           sma_ecc_inc_combination_mode,
                           rank,
                           size,
                           overwrite_previous=False,
                           include_shadow_bool=bool(include_shadow_b),
                           run_mode='keplerian_vane_detumbling',
                           output_frequency_in_seconds_=10,
                           initial_orientation_str='sun_pointing')

runPropagationAnalysis(all_combinations,
                          optical_mode_str,
                          sma_ecc_inc_combination_mode,
                          rank,
                          size,
                          overwrite_previous=False,
                          include_shadow_bool=bool(include_shadow_b),
                          run_mode='vane_detumbling',
                          output_frequency_in_seconds_=10,
                          initial_orientation_str='sun_pointing')
