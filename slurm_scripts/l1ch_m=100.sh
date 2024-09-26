#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=24:00:00
#SBATCH --job-name="l1ch_m=100"
#SBATCH --mail-user=yalan@stanford.edu
#SBATCH --mail-type=END
#SBATCH --output=l1ch_m=100_%j.txt
#SBATCH --error=l1ch_m=100_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --mem=256G
#SBATCH --partition=bigmem
#####################################

# run using sbatch --array=0,10,20,30,40,50,1000 gps_l1.sh

# Load module for Gurobi and Julia (should be most up-to-date version, i.e. 1.7.2)
module load python/3.9

# Change to the directory of script
export SLURM_SUBMIT_DIR=/scratch/users/yalan/decor

# Change to the job directory
cd $SLURM_SUBMIT_DIR

python3 -m scripts.optimize --seed $SLURM_ARRAY_TASK_ID --n 210 --len 10230 --p 6 --iters 1_000_000_000 --method TopKGreedyCodeOptimizer --args num_neighbors=100
