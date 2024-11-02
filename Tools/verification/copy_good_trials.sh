#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Lahiru Copy Good Trials"
#SBATCH --error="./logs/job-%j-lahiru_copy.err"
#SBATCH --output="./logs/job-%j-lahiru_copy.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoexoems &&
python -u lahiru_copy_good_trials.py &&
echo "Done" &&
exit

done
