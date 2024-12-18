#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-mmosmond
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40


module load NiaEnv/2019b
module load gcc
module load NiaEnv/2019b gnu-parallel

source ~/.virtualenvs/myenv/bin/activate

python ComputingDispersalRates.py 

