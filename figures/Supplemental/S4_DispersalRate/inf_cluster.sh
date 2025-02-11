#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-mmosmond
#SBATCH --cpus-per-task=40
#SBATCH --nodes=2

module load CCEnv
module load StdEnv
module load nixpkgs/16.09
module load gcc/7.3.0
module load parallel/20160722

source ~/.virtualenvs/myenv/bin/activate

python inf_cluster.py

