import os
import shutil
import pandas as pd

# Paths
good_trials_file = 'trial_status.csv'  # CSV file with 'subject', 'trial', and 'quality' columns
source_dataset_dir = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Lahiru/'
destination_dir = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/'  # Directory to copy "good" trials

# Load good trials from the CSV file
def load_good_trials(file_path):
    df = pd.read_csv(file_path)
    # Filter only "good" trials and create a set of (subject, trial) tuples
    return set(zip(df[df['quality'] == 'good']['subject'], df[df['quality'] == 'good']['trial']))

# Copy good trial folders to the destination directory
def copy_good_trials(good_trials, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)  # Ensure destination directory exists
    copy_count = 0

    for subject_id in os.listdir(source_dir):
        subject_path = os.path.join(source_dir, subject_id)
        
        # Check if the subject directory exists and iterate through procedures
        if os.path.isdir(subject_path):
            for procedure_id in os.listdir(subject_path):
                procedure_path = os.path.join(subject_path, procedure_id)
                
                # Iterate through trials within the procedure
                for trial_id in os.listdir(procedure_path):
                    trial_path = os.path.join(procedure_path, trial_id)
                    
                    # Check if this trial is marked as "good"
                    if (subject_id, trial_id) in good_trials:
                        dest_trial_path = os.path.join(dest_dir, subject_id, procedure_id, trial_id)
                        shutil.copytree(trial_path, dest_trial_path)
                        print(f"Copied {trial_path} to {dest_trial_path}")
                        copy_count += 1

    print(f"Total {copy_count} good trials copied to {dest_dir}")
# Load the good trials from CSV
good_trials = load_good_trials(good_trials_file)

# Copy the good trial folders
copy_good_trials(good_trials, source_dataset_dir, destination_dir)

