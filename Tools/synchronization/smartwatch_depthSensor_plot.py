import os
import pandas as pd
import matplotlib.pyplot as plt


#THIS CODE GRAPHS SMARTWATCH AND DEPTH SENSOR CODE AND SAVES OUTPUT AS A PNG
#THIS CODE IS MEANT TO BE USED AFTER PULLFILE.SH, WHEN DATA IS IN SCRATCH DIRECTORY
#CAN ALSO BE USED FOR POST SYNCHRONIZATION PLOTTING, IF MAIN DIRECTORY IS ADJUSTED
#this code assumes the directory is date/person/trial/dataFiles
#main folder should be the date directory post file pull. 
input_dir = '06-09-2024' 


#1 IF SYNCHRONIZED DATA, 0 if raw
synchronized_plot = 1

if (synchronized_plot):
    output_dir = os.path.join("sychronized_sw_depthSensor_plots", input_dir)

else:
    output_dir = os.path.join("sw_depthSensor_plots", input_dir)

os.makedirs(output_dir, exist_ok=True)

for person in os.listdir(input_dir):
    person_dir = os.path.join(input_dir, person)
    for intervention in os.listdir(person_dir):
        intervention_dir = os.path.join(person_dir, intervention)
        for trial in os.listdir(intervention_dir):  # Corrected to loop through intervention_dir
            trial_dir = os.path.join(intervention_dir, trial)

            # Create depth and smartwatch folders within each trial
            depth_folder = os.path.join(trial_dir, 'depthSensor_data')
            smartwatch_folder = os.path.join(trial_dir, 'smartwatch_data')

            if synchronized_plot:
                # POST SYNCHRONIZATION CODE
                for f in os.listdir(depth_folder):
                    if f.endswith("depthSensor.csv"):
                        depth_file_path = os.path.join(depth_folder, f)
                        depth_data = pd.read_csv(depth_file_path)
                        print(f"Found depth for {person} {trial}")

                for f in os.listdir(smartwatch_folder):
                    if f.endswith("smartwatch.csv"):
                        sw_file_path = os.path.join(smartwatch_folder, f)
                        sw_data = pd.read_csv(sw_file_path)
                        print(f"Found smartwatch for {person} {trial}")
            else:
                for f in os.listdir(depth_folder):
                    if f.startswith("VL"):
                        depth_file_path = os.path.join(depth_folder, f)
                        depth_data = pd.read_csv(depth_file_path)
                        print(f"Found depth for {person} {trial}")

                for f in os.listdir(smartwatch_folder):
                    if f.startswith("sw_data"):
                        sw_file_path = os.path.join(smartwatch_folder, f)
                        sw_data = pd.read_csv(sw_file_path)
                        print(f"Found smartwatch for {person} {trial}")

            if depth_data.empty:
                print(f"WARNING: missing depth data for {person} {trial}")

            if not depth_data.empty:
                depth_values = depth_data[depth_data.columns[0]].tolist()

                    # Load data from the CSV file
                csv_X_values = sw_data['sw_value_X_Axis'].tolist()
                csv_Y_values = sw_data['sw_value_Y_Axis'].tolist()
                csv_Z_values = sw_data['sw_value_Z_Axis'].tolist()

                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                    # Plot values from the text file
                ax1.plot(range(len(depth_values)), depth_values, linestyle='-', color='b')
                ax1.set_title(f'Values from Depth Camera ({os.path.basename(depth_file_path)})')
                ax1.set_xlabel('Frame')
                ax1.set_ylabel('Value')
                ax1.grid()

                print(f"plotting for {person } {trial}")

                    # Plot values from the CSV file
                ax2.plot(range(len(csv_X_values)), csv_X_values, linestyle='-', color='r', label='X Axis')
                ax2.plot(range(len(csv_Y_values)), csv_Y_values, linestyle='-', color='b', label='Y Axis')
                ax2.plot(range(len(csv_Z_values)), csv_Z_values, linestyle='-', color='g', label='Z Axis')
                ax2.set_title(f'Values from Smart Watch ({os.path.basename(sw_file_path)})')
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Value')
                ax2.grid()
                ax2.legend()

                    
                plt.tight_layout()
                plot_filename = f"{person}_{trial}_plot.png"
                output_file_path = os.path.join(output_dir, plot_filename)  
                plt.savefig(output_file_path)
                plt.close()  

            
    