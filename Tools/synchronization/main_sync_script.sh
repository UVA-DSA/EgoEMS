#!/bin/bash

#SBATCH --job-name=synchronize
#SBATCH --output=/scratch/vht2gm/
#SBATCH --time=01:00:00

# Definitions
root_dir="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
sync_dir="$root_dir/05-09-2024"

python goPro_timestamp_adjuster.py --root_dir "$root_dir" &&
python synchronization-v2.py --root_dir "$root_dir"  &&
python gopro_trimmer.py --root_dir "$root_dir"  &&
python kinect_trimmer.py --root_dir "$root_dir" &&
python sync_clip_merger.py --root_dir "$sync_dir" &&
echo "Synchronization complete" 