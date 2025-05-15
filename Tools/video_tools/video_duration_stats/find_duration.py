import os
import csv
import subprocess
import json

def find_large_mp4_files(folder, size_threshold_mb=100):
    large_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('synced_encoded.mp4'):
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if size_mb > size_threshold_mb:
                    large_files.append(filepath)
    return large_files

def get_video_duration_minutes(filepath):
    try:
        # Use ffprobe to get video duration
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'format=duration',
                '-of', 'json',
                filepath
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        info = json.loads(result.stdout)
        print(f"Duration info for {filepath}: {info}")
        duration_seconds = float(info['format']['duration'])
        return duration_seconds / 60
    except Exception as e:
        print(f"Warning: Failed to get duration for {filepath}: {e}")
        return None

def main(input_folder, output_csv):
    files = find_large_mp4_files(input_folder)
    print(f"Found {len(files)} MP4 files larger than 300MB.")

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filepath', 'Duration_minutes'])

        for file in files:
            duration = get_video_duration_minutes(file)
            if duration is not None:
                writer.writerow([file, round(duration, 2)])

    print(f"CSV written to {output_csv}")

if __name__ == "__main__":

    # get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Find large MP4 files and get their durations.")
    parser.add_argument('--input_folder', type=str, required=True, help="Input folder to search for MP4 files.")
    parser.add_argument('--output_csv', type=str, required=True, help="Output CSV file to write durations.")


    args = parser.parse_args()
    input_folder = args.input_folder
    output_csv = args.output_csv

    main(input_folder, output_csv)
