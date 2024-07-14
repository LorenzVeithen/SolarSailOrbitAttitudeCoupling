from solarSailPropagationFunction import runPropagationAnalysis
from vaneDetumbling_ACS3Model import sail_I
#from mpi4py import MPI
import numpy as np
import itertools
import sys
import random

#random.seed(42)
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
#n_processes = size

# argv[0] is the executable
#optical_model_mode = int(sys.argv[1])  # 0: ACS3 optical model, 1: double-sided  ideal model, 2: single-sided ideal model
#sma_ecc_inc_combination_mode = int(sys.argv[2])    # see below for the combinations
#include_shadow_b = int(sys.argv[3])     #0: False (no shadow), 1: True (with shadow)
#optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][optical_model_mode]

rho_debris = 2700   # kg/m^3
beta_list = [1, 2, 3, 4, 5]
position_on_boom_list = [1, 3, 5, 7]
projectile_diameter = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]) * 1e-3    # m
projectile_mass = (4/3) * np.pi * projectile_diameter**3 * rho_debris
projectile_velocity = np.array([1, 3, 5, 7, 10, 15, 20, 25]) * 1e3  # m/s
projectile_combinations = list(itertools.product(projectile_mass, projectile_velocity))

linear_momentum_list = []
for projectile_comb in projectile_combinations:
    linear_momentum_list.append(projectile_comb[0] * projectile_comb[1])

#TODO: add direction
#TODO: think about what is too much to consider and what is not enough, and a proper justification for these
all_combinations = []
for p in linear_momentum_list:
    for beta in beta_list:
        for direction_id, direction_array in enumerate([np.array([1, 0, 0]), np.array([0, 1, 0])]):
            for position_on_boom in position_on_boom_list:
                impact_body_fixed_position = position_on_boom * direction_array

                momentum_transfered = beta * np.array([0, 0, 1]) * p
                omega_increment = np.linalg.inv(sail_I) * (np.cross(impact_body_fixed_position, momentum_transfered))
                print("------")
                print(p)
                print(np.rad2deg(np.linalg.norm(omega_increment)))
                all_combinations.append((omega_increment[0], omega_increment[1], omega_increment[2]))

print(len(all_combinations))
"""
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
"""