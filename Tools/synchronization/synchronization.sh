#!/bin/bash

#SBATCH --job-name=synchronize
#SBATCH --output=/scratch/vht2gm/
#SBATCH --time=01:00:00


module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

pip install pandas


folders=("05-09-2024" "06-09-2024")

for folder in "${folders[@]}'; do
    echo "Processing folder: $folder"
   
    python synchronization.py  "$folder"

done


