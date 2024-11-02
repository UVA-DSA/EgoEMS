import os
import json
import pandas as pd

# Define paths to folders
good_folder = './good'
all_images_folder = './output'
output_json_file = 'trial_status.json'
output_csv_file = 'trial_status.csv'

# Extract trial information from filenames in the good folder
good_trials = set()
for filename in os.listdir(good_folder):
    if filename.endswith('_sw_depth.png'):
        # Extract subject and trial info from filename
        parts = filename.split('_')
        subject = parts[0]
        trial = parts[1]
        good_trials.add((subject, trial))

# Initialize a list to store the status of each trial
trial_status = []

# Check all images and determine if each is "good" or "bad"
for filename in os.listdir(all_images_folder):
    if filename.endswith('_sw_depth.png'):
        parts = filename.split('_')
        subject = parts[0]
        trial = parts[1]
        quality = 'good' if (subject, trial) in good_trials else 'bad'
        
        # Add each entry as a dictionary for JSON and CSV export
        trial_status.append({'subject': subject, 'trial': trial, 'quality': quality})

# Save the results to a JSON file
with open(output_json_file, 'w') as f:
    json.dump(trial_status, f, indent=4)
print(f"Trial status saved to {output_json_file}")

# Convert to CSV format
trial_status_df = pd.DataFrame(trial_status)
trial_status_df.to_csv(output_csv_file, index=False)
print(f"Trial status saved to {output_csv_file}")
