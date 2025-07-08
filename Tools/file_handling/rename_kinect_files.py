import os
import glob

# Base directory where files are located
base_directory = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"

# Find all incorrectly named files
incorrect_files = glob.glob(f"{base_directory}/**/*_stream_deidentified.mp4", recursive=True)

for file_path in incorrect_files:
    # Split the path into components
    path_parts = file_path.split(os.sep)

    # Extract subject and trial correctly
    try:
        subject = path_parts[-5]  # Subject (e.g., P16, ng2, ms1)
        scenario = path_parts[-4]  # Scenario (e.g., s5, t1, s2)
        scenario = scenario.replace("_","")
        trial = path_parts[-3]    # Trial (e.g., s5, t1)
        print("-*" * 20)
        # Construct correct filename
        correct_filename = f"{subject}_{scenario}_t{trial}_exo_final.mp4"
        correct_filepath = os.path.join(os.path.dirname(file_path), correct_filename)

        # Rename the file
        # os.rename(file_path, correct_filepath)

        # copy the file to the new location
        os.system(f'cp "{file_path}" "{correct_filepath}"')
        print(f"Fixed: {file_path} -> {correct_filepath}")
    
    except IndexError:
        print(f"Skipping (Invalid path structure): {file_path}")

print("✅ Renaming correction completed successfully.")
