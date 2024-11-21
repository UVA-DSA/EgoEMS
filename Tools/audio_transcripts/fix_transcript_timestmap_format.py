import json

# Load the JSON data
transcript_path = '/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/gemini_GX010390_encoded_trimmed_timestamped.json'
with open(transcript_path, "r") as file:
    data = json.load(file)

# Convert m:ss to hh:mm:ss
def convert_to_hhmmss(time_str):
    minutes, seconds = map(int, time_str.split(":"))
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Update timestamps in the JSON data
for entry in data:
    entry["start_t"] = convert_to_hhmmss(entry["start_t"])
    entry["end_t"] = convert_to_hhmmss(entry["end_t"])

# Save the updated JSON
output_path = transcript_path
with open(output_path, "w") as file:
    json.dump(data, file, indent=4)

output_path
