from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from datetime import datetime
import json

# Paths (unchanged from your setup)
transcript_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/gemini_GX010390_encoded_trimmed_timestamped.json'
audio_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/GX010390_encoded_trimmed.mp3'
video_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/GoPro/GX010390_encoded_trimmed_deidentified.mp4'


# transcript_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/ng2/cardiac_arrest/0/audio/gemini_GX010341_encoded_trimmed_timestamped.json'
# audio_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/ng2/cardiac_arrest/0/audio/GX010341_encoded_trimmed.mp3'
# video_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/ng2/cardiac_arrest/0/GoPro/GX010341_encoded_trimmed.mp4'



# Extract metadata
subject = video_path.split("/")[-5]
trial = video_path.split("/")[-3]

print(f"Subject: {subject}, Trial: {trial}")

# Load the transcript data
transcript_data = json.load(open(transcript_path, "r"))

# Define colors for different roles
role_colors = {
    "First Responder 1": "white",
    "First Responder 2": "green",
    "Patient": "purple",
    "AED": "yellow"
}

# Function to convert timestamp to seconds
def time_to_seconds(t):
    dt = datetime.strptime(t, "%H:%M:%S")
    return dt.hour * 3600 + dt.minute * 60 + dt.second

# Load video and audio
video = VideoFileClip(video_path)
audio = AudioFileClip(audio_path)
subtitles = []

height_base = video.h - 100  # Base height for subtitles
line_height = 50  # Vertical spacing between overlapping subtitles

# Keep track of currently active subtitles to avoid overlap
active_subtitles = []

# Create text clips for each transcript entry
for entry in transcript_data:
    start_time = time_to_seconds(entry["start_t"])
    end_time = time_to_seconds(entry["end_t"])
    duration = end_time - start_time
    
    role = entry["Role"]
    text = f"{role}: {entry['Utterance']}"
    color = role_colors.get(role, "white")
    
    # Adjust subtitle position dynamically to avoid overlap
    # Clear finished subtitles
    active_subtitles = [sub for sub in active_subtitles if sub["end_time"] > start_time]
    
    # Calculate y-offset based on active subtitles
    y_offset = height_base - (len(active_subtitles) * line_height)
    active_subtitles.append({"end_time": end_time})
    
    # Create the TextClip
    text_clip = TextClip(
        text, fontsize=44, color=color, font="Arial-Bold", method="caption", size=(video.w * 0.8, None)
    ).set_position(("center", y_offset)).set_duration(duration).set_start(start_time)
    
    subtitles.append(text_clip)

# Combine video with all subtitle text clips
final_video = CompositeVideoClip([video] + subtitles)

# Add audio to the video
final_video_with_audio = final_video.set_audio(audio)

# Save the final video with audio and subtitles
video_name = video_path.split("/")[-1]
output_video_name = video_name.replace(".mp4", "_with_subtitles.mp4")
output_video_path = f"./output/{subject}_{trial}_{output_video_name}"
final_video_with_audio.write_videofile(output_video_path, codec="libx264", fps=29.97)

print(f"Final video with audio and subtitles saved to {output_video_path}")
