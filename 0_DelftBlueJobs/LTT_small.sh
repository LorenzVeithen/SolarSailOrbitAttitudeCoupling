#!/bin/bash

#SBATCH --job-name="LTT_small"
#SBATCH --time=07:00:00
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-ae-space

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

module load 2023r1
module load openmpi
module load python
module load miniconda3
module load openssh
module load git

# Activate conda, run job, deactivate conda
conda activate <name-of-my-conda-environment>
srun python myscript.py
conda deactivate
srun python calculate_pi.py > pi.log
