#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS Synchronization Task"
#SBATCH --error="./logs/job-%j-egoexoems_sync_task.err"
#SBATCH --output="./logs/job-%j-egoexoems_sync_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate cogems &&

# Set the root directory and day variables
root_dir="/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
day="05-09-2024"

# Step 1: Adjust the timestamp offset of GoPro
echo "Adjusting GoPro timestamp offset..."
python goPro_timestamp_adjuster.py "$root_dir"
if [ $? -ne 0 ]; then
    echo "Error adjusting GoPro timestamp offset."
    exit 1
fi

# Step 2: Generate synchronization metadata
echo "Generating synchronization metadata..."
python synchronization-v2.py "$root_dir" "$day"
if [ $? -ne 0 ]; then
    echo "Error generating synchronization metadata."
    exit 1
fi

# Step 3: Convert Kinect frame rate to 29.97
echo "Converting Kinect frame rate to 29.97 FPS..."
python kinect_fps_converter.py "$root_dir"
if [ $? -ne 0 ]; then
    echo "Error converting Kinect frame rate."
    exit 1
fi

# Step 4: Trim GoPro recordings
echo "Trimming GoPro recordings..."
python gopro_trimmer.py "$root_dir"
if [ $? -ne 0 ]; then
    echo "Error trimming GoPro recordings."
    exit 1
fi

# Step 5: Trim Kinect recordings
echo "Trimming Kinect recordings..."
python kinect_trimmer.py "$root_dir"
if [ $? -ne 0 ]; then
    echo "Error trimming Kinect recordings."
    exit 1
fi

# Step 6: Create a side-by-side preview of synchronized GoPro and Kinect
echo "Creating side-by-side preview of synchronized videos..."
python sync_clip_merger.py "$root_dir" "$day"
if [ $? -ne 0 ]; then
    echo "Error creating side-by-side preview."
    exit 1
fi

echo "Synchronization steps completed successfully."