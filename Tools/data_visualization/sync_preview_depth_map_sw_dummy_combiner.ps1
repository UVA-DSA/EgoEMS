# Define path to ffmpeg (if not in system PATH)
$ffmpegPath = "ffmpeg"

# Define input video paths (replace with full paths if needed)
$video1 = "wa1_t5_synched_preview.mp4"
$video2 = "wa1_t5_depth.mp4"
$video3 = "wa1_t5_smartwatch_final.mp4"
$video4 = "wa1_t5_depth_sensor.mp4"

# Define output video path
$outputVideo = "output_video.mp4"

# Run ffmpeg command with filter complex for arranging videos with matched height
& $ffmpegPath -i $video1 -i $video2 -i $video3 -i $video4 `
-filter_complex "[0:v]scale=1920:540[top]; `
[1:v]scale=640:540[v2]; `
[2:v]scale=640:540[v3]; `
[3:v]scale=640:540[v4]; `
[v2][v3][v4]hstack=3[bottom]; `
[top][bottom]vstack" `
-c:v libx264 -pix_fmt yuv420p $outputVideo

Write-Output "Video combination complete! Output saved as $outputVideo"
