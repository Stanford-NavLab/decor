#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=24:00:00
#SBATCH --job-name="gal_greedy"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=gal_greedy_%j.txt
#SBATCH --error=gal_greedy_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=4G
#SBATCH --partition=normal
#####################################

# run using sbatch --array=0,10,20,30,40,50,1000 gps_l1.sh

# Load module for Gurobi and Julia (should be most up-to-date version, i.e. 1.7.2)
module load python/3.9

# Change to the directory of script
export SLURM_SUBMIT_DIR=/scratch/users/yalan/decor

# Change to the job directory
cd $SLURM_SUBMIT_DIR

python3 -m scripts.optimize --seed $SLURM_ARRAY_TASK_ID --n 100 --len 4092 --p 6 --iters 1_000_000_000 --method GreedyCodeOptimizer