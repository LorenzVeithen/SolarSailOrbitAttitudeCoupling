#!/bin/bash

#SBATCH --job-name="LTT_small"
#SBATCH --time=00:15:00
#SBATCH --ntasks=3
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
conda activate LV-tudat-bundle
srun python scatch/lveithen/SourceCode_24_06_2024/LongTermTumblingAnalysis/LTT_MPI.py > scratch/lveithen/SourceCode_24_06_2024/0_DelftBlue_Jobs/LTT_small.log
conda deactivate


