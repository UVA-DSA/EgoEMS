#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Deidentify EgoExoEMS GoPro dataset."
#SBATCH --error="tunnel.err"
#SBATCH --output="tunnel.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge &&
module load anaconda

conda env create --file='/scratch/cqa3ym/repos/EgoExoEMS/Tools/deidentification/egoblur/EgoBlur/environment.yaml'
conda activate ego_blur

script_path='/scratch/cqa3ym/repos/EgoExoEMS/Tools/deidentification/egoblur/EgoBlur/script/demo_ego_blur.py'
model_path='/scratch/cqa3ym/repos/EgoExoEMS/Tools/deidentification/egoblur/EgoBlur/model/ego_blur_face.jit'

for file in /standard/UVA-DSA/NIST\ EMS\ Project\ Data/CognitiveEMS_Datasets/North_Garden/Final/*/*/*/*/GoPro/*.mp4
do
  printf "$(date): Deidentifying %s\n" "$file"
  filename=$(basename "$file")
 	filepath=$(dirname "$file")
	new_filename="${filename%.mp4}_deidentified.mp4"
	output_path="$filepath/$new_filename"
	printf "Output path: %s\n" "$output_path"
	python $script_path --face_model_path $model_path --input_video_path "$file" --output_video_path "$output_path"
done
