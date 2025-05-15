import os
import shutil
import pandas as pd
from collections import defaultdict

# === CONFIGURATION ===
mapping_file = "./opvrs_data_mappings.csv"   # Path to CSV
is_tab_separated = False
dest_base = "/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/OPVRS/organized/"

# === READ MAPPING ===
sep = '\t' if is_tab_separated else ','
df = pd.read_csv(mapping_file, sep=sep)

# Drop rows with missing values
df = df.dropna(subset=['Filepath', 'Unique_ID', 'Scenario', 'Smartwatch'])

# Normalize scenario names
df['Scenario'] = df['Scenario'].str.lower().str.replace(" ", "_")


def organize_gopro_files():
    # Round-robin counter for assigning trial_num (0–3) per subject+scenario
    trial_counters = defaultdict(int)

    # === COPY FILES ===
    for idx, row in df.iterrows():
        src_mp4 = row['Filepath'].strip()
        subject_id = row['Unique_ID'].strip()
        category = row['Scenario'].strip()
        smartwatch = row['Smartwatch'].strip()
        
        smartwatch = smartwatch.split(".")[0] # Remove the file extension if present

        # remove - and () from category
        category = category.replace("-", "").replace("(", "").replace(")", "")


        if not os.path.isfile(src_mp4):
            print(f"[WARN] MP4 not found: {src_mp4}")
            continue

        # Determine trial number
        key = (subject_id, category)
        trial_num = str(trial_counters[key] % 4)
        trial_counters[key] += 1

        mp4_name = os.path.basename(src_mp4)
        base_number = mp4_name[-10:-4]  # e.g., "010032"
        src_dir = os.path.dirname(src_mp4)

        # Construct correct filenames
        file_variants = {
            "MP4": f"GX{base_number}.MP4",
            "THM": f"GX{base_number}.THM",
            "LRV": f"GL{base_number}.LRV",
            "csv": f"GX{base_number}.csv",
        }

        dest_dir = os.path.join(dest_base, subject_id, category, trial_num, "GoPro")
        os.makedirs(dest_dir, exist_ok=True)

        for ext, filename in file_variants.items():
            full_src = os.path.join(src_dir, filename)
            if os.path.isfile(full_src):

                # check if the file is already in the destination
                dest_file = os.path.join(dest_dir, filename)
                if os.path.isfile(dest_file):
                    print(f"[EXISTS] {filename} already exists in {dest_dir}")
                    continue
                shutil.copy2(full_src, dest_dir)
                print(f"[OK] Copied {filename} → {dest_dir}")
            else:
                print(f"[MISSING] {filename} in {src_dir}")

    print("✅ All done.")


def organize_smartwatch_files():
    
    smartwatch_base_dir = "/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/OPVRS/RAW/SW/"
    trial_counters = defaultdict(int)
    
    copy_count = 0
    
    # === COPY FILES ===
    for idx, row in df.iterrows():
        print("-" * 20)
        src_mp4 = row['Filepath'].strip()
        subject_id = row['Unique_ID'].strip()
        category = row['Scenario'].strip()
        smartwatch = row['Smartwatch'].strip()
        
        smartwatch = smartwatch.split(".")[0] # Remove the file extension if present

        # remove - and () from category
        category = category.replace("-", "").replace("(", "").replace(")", "")
        
        # replace non_cardiac with noncardiac
        category = category.replace("_non_cardiac", "noncardiac")
        category = category.replace("_hypoglycemic", "hypoglycemic")
        
        #  extract responder type from src_mp4
        respoder_type = src_mp4.split("/")[-2][2:]  # e.g., "01" or "02" or "03"
        
        print(f"Processing smartwatch dir {smartwatch}  for {respoder_type} responder with {subject_id} in {category}...")
        
        
        smartwatch_file_dir = os.path.join(smartwatch_base_dir, smartwatch)
        smartwatch_file = os.path.join(smartwatch_file_dir, f"synchronized_smartwatch_{respoder_type}.csv")
        
        # Check if the smartwatch file exists
        if not os.path.isfile(smartwatch_file):
            print(f"[WARN] Smartwatch file not found: {smartwatch_file}")
            continue

        # Determine trial number
        key = (subject_id, category)
        trial_num = str(trial_counters[key] % 4)
        trial_counters[key] += 1
        
        dest_dir = os.path.join(dest_base, subject_id, category, trial_num, "smartwatch_data")
        
        # copy the smartwatch file to the destination directory
        dest_file = os.path.join(dest_dir, f"synchronized_smartwatch_{respoder_type}.csv")
        if os.path.isfile(dest_file):
            print(f"[EXISTS] {smartwatch_file} already exists in {dest_dir}")
            continue
        shutil.copy2(smartwatch_file, dest_dir)
        print(f"[OK] Copied {smartwatch_file} → {dest_dir}")
        copy_count += 1
        
    print(f"✅ {copy_count} smartwatch files copied.")

def delete_empty_trials():
        
    # Base path to the OPVRS folders
    base_dir = "/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/OPVRS/organized/"  # CHANGE THIS

    valid_trials = 0
    deleted_trials = 0
    # Iterate over all opvrs_* folders
    for opvrs_folder in os.listdir(base_dir):
        opvrs_path = os.path.join(base_dir, opvrs_folder)
        if not os.path.isdir(opvrs_path):
            continue

        # Go inside 'chest_pain' and 'stroke' if they exist
        for scenario in ['chest_pain', 'stroke', 'stroke_unresponsive', 'stroke_hypoglycemic', 'chest_pain_noncardiac']:
            scenario_path = os.path.join(opvrs_path, scenario)
            if not os.path.isdir(scenario_path):
                continue

            # Check each numbered trial folder
            for trial in os.listdir(scenario_path):
                trial_path = os.path.join(scenario_path, trial)
                gopro_path = os.path.join(trial_path, "GoPro")
                print("*-" * 20)

                if os.path.isdir(trial_path) and os.path.isdir(gopro_path):
                    # If GoPro folder exists and is empty
                    if not os.listdir(gopro_path):
                        print(f"[DELETE] {trial_path}")
                        deleted_trials += 1
                        # shutil.rmtree(trial_path)
                    else:
                        print(f"[KEEP] {trial_path} is not empty.")
                        valid_trials += 1

                print("*-" * 20)

    print(f"✅ {valid_trials} valid trials found.")
    print(f"!!! {deleted_trials} empty trials deleted.")
    
    
    
def delete_empty_scenarios():
    base_dir = "/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/OPVRS/organized/"  # CHANGE THIS

    valid_scenarios = 0
    deleted_scenarios = 0

    for opvrs_folder in os.listdir(base_dir):
        opvrs_path = os.path.join(base_dir, opvrs_folder)
        if not os.path.isdir(opvrs_path):
            continue

        for scenario in ['chest_pain', 'stroke', 'stroke_unresponsive', 'stroke_hypoglycemic', 'chest_pain_noncardiac']:
            scenario_path = os.path.join(opvrs_path, scenario)
            if not os.path.isdir(scenario_path):
                continue

            all_trials_empty = True
            for trial in os.listdir(scenario_path):
                trial_path = os.path.join(scenario_path, trial)
                gopro_path = os.path.join(trial_path, "GoPro")

                if os.path.isdir(trial_path) and os.path.isdir(gopro_path):
                    if os.listdir(gopro_path):  # GoPro has files
                        all_trials_empty = False
                        break

            if all_trials_empty:
                print(f"[DELETE SCENARIO] {scenario_path}")
                deleted_scenarios += 1
                shutil.rmtree(scenario_path)
            else:
                print(f"[KEEP SCENARIO] {scenario_path}")
                valid_scenarios += 1

    print(f"✅ {valid_scenarios} scenarios kept.")
    print(f"!!! {deleted_scenarios} scenarios deleted.")
    
if __name__ == "__main__":
    print("=== OPVRS Data Organization ===")
    print(f"Mapping file: {mapping_file}")
    # Organize GoPro files
    # organize_gopro_files()

    # Organize Smartwatch files
    # organize_smartwatch_files()
    
    # Delete empty trials
    # delete_empty_trials()
    
    # Delete empty scenarios
    # delete_empty_scenarios()