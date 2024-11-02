import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the subject info from JSON
subjects = json.load(open('./lahiru_subjects.json'))

# List of IRB verified subject IDs
verified_irb_subjects = [subject['subject_id'] for subject in subjects['irb_verified_subjects']]

dataset_dir = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Lahiru/'

plot_path = "./output/"
os.makedirs(plot_path, exist_ok=True)

trial_count = 0
# Iterate through each subject in the dataset directory
# Iterate through each subject in the dataset directory
for subject_id in os.listdir(dataset_dir):
    subject_path = os.path.join(dataset_dir, subject_id)
    
    # Check if the subject is in the verified IRB list
    if subject_id in verified_irb_subjects and os.path.isdir(subject_path):
        # Iterate through each procedure within the subject
        for procedure_id in os.listdir(subject_path):
            procedure_path = os.path.join(subject_path, procedure_id)
            
            # Iterate through each trial for the procedure
            for trial_id in os.listdir(procedure_path):
                trial_path = os.path.join(procedure_path, trial_id)
                
                # Paths to smartwatch and depth sensor data files
                smartwatch_path = os.path.join(trial_path, 'smartwatch_data', 'sync_smartwatch.csv')
                depth_sensor_path = os.path.join(trial_path, 'distance_sensor_data', 'sync_depth_sensor.csv')
                
                # Check if both smartwatch and depth sensor files exist
                if os.path.isfile(smartwatch_path) and os.path.isfile(depth_sensor_path):

                    print("[INFO] Plotting for", subject_id, trial_id)
                    trial_count += 1
                    # Load smartwatch and depth sensor data
                    smartwatch_data = pd.read_csv(smartwatch_path)
                    depth_data = pd.read_csv(depth_sensor_path)
                    
                    # Generate time axis in seconds assuming a 30Hz sampling rate
                    time_seconds = [i / 30 for i in range(len(depth_data))]
                    
                    # Plotting
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot depth sensor data
                    ax1.plot(time_seconds, depth_data['depth_value'], color='blue')
                    ax1.set_title(f'Depth Sensor Data - {subject_id} Trial {trial_id}')
                    ax1.set_xlabel('Time (Seconds)')
                    ax1.set_ylabel('Depth Value')
                    
                    # Plot smartwatch data (assuming X, Y, Z values over time)
                    ax2.plot(time_seconds, smartwatch_data['sw_value_X_Axis'], label='X Axis', color='red')
                    ax2.plot(time_seconds, smartwatch_data['sw_value_Y_Axis'], label='Y Axis', color='green')
                    ax2.plot(time_seconds, smartwatch_data['sw_value_Z_Axis'], label='Z Axis', color='blue')
                    ax2.set_title(f'Smartwatch Data - {subject_id} Trial {trial_id}')
                    ax2.set_xlabel('Time (Seconds)')
                    ax2.set_ylabel('Acceleration (Axis Values)')
                    ax2.legend()
                    
                    # Save the plot in the trial directory
                    save_path = os.path.join(plot_path, f"{subject_id}_{trial_id}_sw_depth.png")
                    plt.savefig(save_path)
                    plt.close(fig)
                    print(f"Plot saved for {subject_id} trial {trial_id} at {save_path}")

print(f"Plotted {trial_count} trials")