# # Define path to ffmpeg (if not in system PATH)
# $ffmpegPath = "ffmpeg"

# # Define input video paths (replace with full paths if needed)
# $video1 = "ng2_t0_ego.mp4"
# $video2 = "ng2_t0_exo.mp4"
# $video3 = "ng2_t0_depth.mp4"
# $video4 = "ng2_t0_smartwatch_final.mp4"
# $video5 = "ng2_t0_depth_sensor.mp4"

# # Define output video path
# $outputVideo = "ng2_t0_modalities_final.mp4"

# # Run ffmpeg command with filter complex for arranging videos
# & $ffmpegPath -i $video1 -i $video2 -i $video3 -i $video4 -i $video5 `
# -filter_complex "[0:v]scale=960:540[v1]; `  # Scale video1
# [1:v]scale=960:540[v2]; `  # Scale video2
# [2:v]scale=640:360[v3]; `  # Scale video3
# [3:v]scale=640:360[v4]; `  # Scale video4
# [4:v]scale=640:360[v5]; `  # Scale video5
# [v1][v2]hstack=2[top]; `  # Stack video1 and video2 horizontally (top row)
# [v3][v4][v5]hstack=3[bottom]; `  # Stack video3, video4, video5 horizontally (bottom row)
# [top][bottom]vstack=2[layout]" `  # Stack top and bottom rows vertically
# -map "[layout]" -c:v libx264 -pix_fmt yuv420p $outputVideo

# Write-Output "Video combination complete! Output saved as $outputVideo"

# Define path to ffmpeg (if not in system PATH)
$ffmpegPath = "ffmpeg"

# Root path where videos are stored
$rootPath = "F:\repos\EgoExoEMS\TestData\ng2_t0\videos_to_combine\"

# Define input video paths using Join-Path
$video1 = Join-Path $rootPath "ng2_0_GX010341_encoded_trimmed_with_subtitles.mp4"
$video4 = Join-Path $rootPath "ng2_t0_smartwatch.mp4"
$video5 = Join-Path $rootPath "ng2_t0_depth_sensor.mp4"

# Run ffmpeg command with filter complex for arranging videos horizontally
& $ffmpegPath -i $video1 -i $video4 -i $video5 -filter_complex "[0:v]scale=640:360[v1];[1:v]scale=640:360[v4];[2:v]scale=640:360[v5];[v1][v4][v5]hstack=inputs=3[layout]" -map "[layout]" -c:v libx264 -pix_fmt yuv420p $outputVideo

# Define output video path (saved in the same root folder)
$outputVideo = Join-Path $rootPath "ng2_t0_modalities_final.mp4"
Write-Output "Video combination complete! Output saved as $outputVideo"
