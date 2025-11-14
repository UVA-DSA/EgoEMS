#!/usr/bin/env bash
#SBATCH --job-name="Audio Extraction Task"
#SBATCH --error="./logs/job-%j-egoems_audio_extraction_task.err"
#SBATCH --output="./logs/job-%j-egoems_audio_extraction_task.output"
#SBATCH --partition="standard"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

module purge
module load miniforge
source /home/cjh9fw/.bashrc
echo "[INFO] Running on node: $HOSTNAME"
conda activate egoexoems
module load ffmpeg

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
input_dir="/standard/UVA-DSA/NIST EMS Project Data/EgoEMS_AAAI2026/cars_1/chest_pain/0/ego/"

# ─── BUILD THE ARRAY ────────────────────────────────────────────────────────────
# Use `find … -print0` + mapfile to get a proper NUL‑delimited list,
# all in the main shell so $count survives.
mapfile -d '' files < <(
  find "$input_dir" -type f -name "*ego_rgb_final.mp4" -print0
)

# ─── PROCESS ────────────────────────────────────────────────────────────────────
count=0
echo "==============================================="
echo "Found ${#files[@]} videos to process in:"
echo "  $input_dir"
echo "==============================================="

for raw_path in "${files[@]}"; do
  # 1) strip stray carriage returns
  video_file="${raw_path//$'\r'/}"

  # 2) sanity‑check
  if [[ ! -f "$video_file" ]]; then
    echo "⚠️  Skipping (not found): '$video_file'"
    continue
  fi

  # 3) derive audio folder
  #    we know each MP4 lives in …/GoPro, so removing that is safer than dirname×2
  video_dir="${video_file%/*}"        # …/GoPro
  parent_dir="${video_dir%/GoPro}"    # strip the trailing /GoPro
  audio_dir="$parent_dir/audio"
  base_name="$(basename "$video_file" .mp4)"
  audio_file="$audio_dir/${base_name}.wav"

  echo
  echo "Processing: '$video_file'"
  echo " → will write: '$audio_file'"

  mkdir -p "$audio_dir"
  ffmpeg -y \
    -i "$video_file" \
    -vn \
    -acodec pcm_s16le \
    -ar 44100 \
    -ac 2 \
    "$audio_file"

  if [[ $? -eq 0 ]]; then
    echo "✅  Extracted: $audio_file"
  else
    echo "❌  Error on: $video_file" >&2
  fi

  ((count++))
  echo "Processed so far: $count"
  echo "-----------------------------------------------"
done

echo
echo "==============================================="
echo "Done!  Total succeeded: $count / ${#files[@]}"
echo "==============================================="
