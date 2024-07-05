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


omega_list = [-85, -70, -55, -40, -30, -20, -10, 85, 70, 55, 40, 30, 20, 10]

all_combinations = list(itertools.product(omega_list, omega_list, [0]))
sampled_combinations = random.sample(all_combinations, 250)
for omega in omega_list:
    if ((omega, omega, omega) not in sampled_combinations):
        sampled_combinations.append((omega, omega, omega))


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
                           output_frequency_in_seconds_=10)

runPropagationAnalysis(all_combinations,
                          optical_mode_str,
                          sma_ecc_inc_combination_mode,
                          rank,
                          size,
                          overwrite_previous=False,
                          include_shadow_bool=bool(include_shadow_b),
                          run_mode='vane_detumbling',
                          output_frequency_in_seconds_=10)
