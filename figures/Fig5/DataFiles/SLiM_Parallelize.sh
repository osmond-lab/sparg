#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-mmosmond
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=20
#SBATCH --nodes=1

module load NiaEnv
module load gcc
module load NiaEnv/2019b gnu-parallel

srun="srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=2"

echo "First Run Started" 

parallel="parallel --delay 0.2 -j 20 --joblog runtask_SpARG_VaryInt.log --resume"

$parallel ~/scratch/SpARG_Simulations/SLiM_build/slim -d SIGMA_int={1} -d SIGMA_comp=1 -d rep={2} simple_space.slim ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ::: 1 2 3 4 5

echo "Second Run Started"

parallel="parallel --delay 0.2 -j 20 --joblog runtask_SpARG_VaryComp.log --resume"

$parallel ~/scratch/SpARG_Simulations/SLiM_build/slim -d SIGMA_int=1 -d SIGMA_comp={1} -d rep={2} simple_space.slim ::: 0.5 1.0 2.0 4.0 6.0 8.0 10.0 ::: 1 2 3 4 5

echo "Done" 

