#!/bin/bash

#SBATCH --job-name="LTT_SP_1_0"
#SBATCH --time=15:00:00
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=12GB
#SBATCH --account=education-ae-msc-ae

module load 2023r1
module load openmpi
module load python
module load miniconda3
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

export PYTHONPATH="/scratch/lveithen/SourceCode":$PYTHONPATH

# Activate conda, run job, deactivate conda
conda activate LV-tudat-bundle
srun python /scratch/lveithen/SourceCode/LongTermTumblingAnalysis/LTT_sun_pointing_MPI.py 1 0 > LTT_large_optical_1_combination_0.log
conda deactivate
