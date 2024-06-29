#!/bin/bash

#SBATCH --job-name="LTT_small"
#SBATCH --time=03:00:00
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=10GB
#SBATCH --account=education-ae-msc-ae

module load 2023r1
module load openmpi
module load python
module load miniconda3
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

export PYTHONPATH="/scratch/lveithen/SourceCode_25_06_2024":$PYTHONPATH

# Activate conda, run job, deactivate conda
conda activate LV-tudat-bundle
srun python /scratch/lveithen/SourceCode_25_06_2024/LongTermTumblingAnalysis/LTT_MPI.py > LTT_small.log
conda deactivate
