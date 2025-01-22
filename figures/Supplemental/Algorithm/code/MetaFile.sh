#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=def-mmosmond
#SBATCH --cpus-per-task=40
#SBATCH --nodes=15
module load CCEnv
module load StdEnv
module load nixpkgs/16.09
module load gcc/7.3.0
module load parallel/20160722

srun="srun --exclusive -N1 -n1 --cpus-per-task 40"

parallel="parallel --delay 0.2 -j 20 --joblog runtask.log --resume"

source ~/.virtualenvs/myenv/bin/activate


$parallel $srun python benchmark.py ${1} ${2} ${3} ::: '10', '40', '70', '100', '150', '200', '250','300','350', '400','450','500','700','1000','4000','7000' ::: '1000' '3000' '5000' ::: '0' '1' '2'

