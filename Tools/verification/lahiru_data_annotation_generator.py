import os
import json
import pandas as pd

# Load good trials from CSV or JSON file
def load_good_trials(file_path):
    good_trials = set()
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Create a set of (subject, trial) tuples for entries marked as "good"
        good_trials = set(zip(df[df['quality'] == 'good']['subject'], df[df['quality'] == 'good']['trial']))
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Assuming the JSON structure also uses (subject, trial) pairs as keys or identifiers
            good_trials = {(entry['subject'], entry['trial']) for entry in data if entry['quality'] == 'good'}
    return good_trials

# Load all keystep options from the provided JSON
def load_options(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    options = data['attribute']['1']['options']
    return options

# Generate annotation.json structure
def create_annotation_file(time_seconds, options, video_file_name='video.mp4'):
    annotation_structure = {
        "project": {
            "pid": "__VIA_PROJECT_ID__",
            "rev": "__VIA_PROJECT_REV_ID__",
            "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
            "pname": "Unnamed VIA Project",
            "creator": "VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)",
            "created": 1730137889755,
            "vid_list": ["1"]
        },
        "metadata": {
            "1_metadata": {
                "vid": "1",
                "flg": 0,
                "z": [0, time_seconds - 5],  # Start at 0, end at time_seconds - 5
                "xy": [],
                "av": {"1": "chest_compressions"}
            }
        },
        "attribute": {
            "1": {
                "aname": "TEMPORAL-SEGMENTS",
                "type": 4,
                "options": options,
                "default_option_id": ""
            }
        },
        "file": {
            "1": {"fid": "1", "fname": video_file_name, "type": 4, "loc": 1, "src": ""}
        },
        "view": {"1": {"fid_list": ["1"]}}
    }
    return annotation_structure

# Directory paths
dataset_dir = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Lahiru/'
good_trials_file = './trial_status.csv'  # or 'trial_status.json'
options_file = './via_project_28Oct2024_14h04m03s.json'

# Load the good trials
good_trials = load_good_trials(good_trials_file)

# Load the options for keysteps
keystep_options = load_options(options_file)

flag = False
# Iterate through each subject in the dataset directory
for subject_id in os.listdir(dataset_dir):
    subject_path = os.path.join(dataset_dir, subject_id)
    
    # Check if the subject directory exists and iterate through procedures
    if os.path.isdir(subject_path):
        for procedure_id in os.listdir(subject_path):
            procedure_path = os.path.join(subject_path, procedure_id)
            
            # Iterate through trials within the procedure
            for trial_id in os.listdir(procedure_path):
                trial_path = os.path.join(procedure_path, trial_id)
                
                # Check if this subject and trial pair is marked as "good"
                if (subject_id, trial_id) in good_trials:
                    # Locate the GoPro folder within the trial directory
                    gopro_folder = os.path.join(trial_path, 'gopro')

                    # find a video that has "_encoded_trimmed.mp4" in the name inside GoPro folder
                    video_file_name = None
                    for file in os.listdir(gopro_folder):
                        if "_encoded_trimmed.mp4" in file:
                            video_file_name = file
                            break

                    print(f"Creating annotation file for {subject_id} {trial_id} at {gopro_folder}")
                    print(f"Video file name: {video_file_name}")

                    depth_sensor_path = os.path.join(trial_path, 'distance_sensor_data', 'sync_depth_sensor.csv')
                    depth_data = pd.read_csv(depth_sensor_path)

                    time_seconds = (len(depth_data) / 30) - 5 # 5 seconds offset to be safe
                    annotation_data = create_annotation_file(time_seconds, keystep_options, video_file_name)

                    video_file_name = video_file_name.split('.')[0]
                    
                    # Save annotation.json in the GoPro folder
                    annotation_file_path = os.path.join(gopro_folder, f'{video_file_name}_annotation.json')
                    with open(annotation_file_path, 'w') as f:
                        json.dump(annotation_data, f, indent=4)
                    print(f"Annotation file created for {subject_id} {trial_id} at {annotation_file_path}")
