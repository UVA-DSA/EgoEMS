import json
import os


def add_timestamps_to_transcript(input_file, output_file):
    # load json data
    data = []
    with open(input_file) as f:
        data = json.load(f)

    # Adding empty start and end times
    for entry in data:
        entry["start_t"] = ""
        entry["end_t"] = ""

    # Print or save the modified JSON
    print(json.dumps(data, indent=4))

    # Optionally save to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    root_dir = "/standard/storage/EgoExoEMS_CVPR2025/Dataset/Final"
    total_files = 0
    # iterate recursively through all files in the root directory
    for root, dirs, files in os.walk(root_dir):
        if "audio" in root:
            print("*"*10, "="*10, "*"*10)
            print(f"[INFO] Processing audio: {root}")

            trial_path = "/".join(root.split("/")[:-1])
            print(f"[INFO] Trial path: {trial_path}")

            # new depthsensor folder

            gemini_transcript_path = None
            for file in files:
                if file.startswith("gemini_") and file.endswith("_encoded_trimmed.json"):
                        gemini_transcript_path = os.path.join(root, file)

            if(gemini_transcript_path):
                total_files += 1
                print("Originial audio transcript: ",gemini_transcript_path)

                # new depthsensor file
                timestamped_gemini_transcript_path = gemini_transcript_path.replace(".json", "_timestamped.json")
                timestamped_gemini_transcript_path = os.path.join(root, timestamped_gemini_transcript_path)
                
                print("Timestamped audio transcript: ",timestamped_gemini_transcript_path)

                add_timestamps_to_transcript(gemini_transcript_path, timestamped_gemini_transcript_path)

                # break

            print("*"*10, "="*10, "*"*10)

    print(f"Total files processed: {total_files}")
