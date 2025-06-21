import os
import glob

# Base directory where files are located
base_directory = "/standard/UVA-DSA/NIST EMS Project Data/DataCollection_Spring_2025/CARS/organized/"

# Find all incorrectly named files
incorrect_files = glob.glob(f"{base_directory}/**/*_synced_720p_gsam2_deidentified.mp4", recursive=True)

counter = 0

for file_path in incorrect_files:
    # Split the path into components
    path_parts = file_path.split(os.sep)

    # Extract subject and trial correctly
    try:
        print("-" * 40)
        subject = path_parts[-5]  # Subject (e.g., P16, ng2, ms1)
        # for cars and opvrs only

        if("cars" in subject or "opvrs" in subject):
            subject = subject.replace("_", "")  # Remove underscores for CARS and OPVRS subjects

    
        trial = path_parts[-3]    # Trial (e.g., s5, t1)

        # Construct correct filename
        correct_filename = f"{subject}_t{trial}_synced_720p_gsam2_deidentified.mp4"
        correct_filepath = os.path.join(os.path.dirname(file_path), correct_filename)

        print(f"Renaming: {file_path} -> {correct_filepath}")

        counter += 1

        # Rename the file
        os.rename(file_path, correct_filepath)
        print("-" * 40)

    
    except IndexError:
        print(f"Skipping (Invalid path structure): {file_path}")

# Print the total number of files renamed
print(f"Total files renamed: {counter}")
print("✅ Renaming correction completed successfully.")
