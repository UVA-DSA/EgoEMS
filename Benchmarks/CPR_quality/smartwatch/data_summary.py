import pandas as pd
import re
import os
import numpy as np

data = []
splits = 4
output_name = 'averages_output.csv'

# Loop over all files
for i in range(1, splits + 1):
    file_path = f'./logs/debug_validate_log_split_{i}.txt'

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Use regex to extract values
                participant_match = re.search(r"\['(.*?)'\]", line)
                trial_match = re.search(r"\['.*?'\],\['(.*?)'\]", line)
                gt_depth_match = re.search(r"GT_Depth:\[(.*?)\]", line)
                pred_depth_match = re.search(r"Pred_Depth:(\d+\.\d+)", line)
                depth_error_match = re.search(r"Depth_error:(\d+\.\d+)", line)
                gt_cpr_rate_match = re.search(r"GT_CPR_rate:\[(.*?)\]", line)
                pred_cpr_rate_match = re.search(r"Pred_CPR_rate:\[(\d+\.\d+)\]", line)
                cpr_rate_error_match = re.search(r"CPR_rate_error:(\d+\.\d+)", line)
                anonymous_pred_cpr_rate_match = re.search(r"anonymous_Pred_CPR_rate:\[(\d+\.\d+)\]", line)
                anonymous_low_pass_pred_cpr_rate_match = re.search(r"Low_Pass_anonymous_Pred_CPR_rate:\[(\d+\.\d+)\]", line)


                # Extracted data
                participant = participant_match.group(1) if participant_match else None
                trial = trial_match.group(1) if trial_match else None
                gt_depth = float(gt_depth_match.group(1)) if gt_depth_match else None
                pred_depth = float(pred_depth_match.group(1)) if pred_depth_match else None
                depth_error = float(depth_error_match.group(1)) if depth_error_match else None
                gt_cpr_rate = float(gt_cpr_rate_match.group(1)) if gt_cpr_rate_match else None
                pred_cpr_rate = float(pred_cpr_rate_match.group(1)) if pred_cpr_rate_match else None
                cpr_rate_error = float(cpr_rate_error_match.group(1)) if cpr_rate_error_match else None
                anonymous_cpr_rate = float(anonymous_pred_cpr_rate_match.group(1)) if anonymous_pred_cpr_rate_match else None
                anonymous_low_pass_cpr_rate = float(anonymous_low_pass_pred_cpr_rate_match.group(1)) if anonymous_low_pass_pred_cpr_rate_match else None
                anonymous_cpr_rate_error = anonymous_cpr_rate - gt_cpr_rate if anonymous_cpr_rate is not None and gt_cpr_rate is not None else None
                anonymous_low_pass_cpr_rate_error = anonymous_low_pass_cpr_rate - gt_cpr_rate if anonymous_low_pass_cpr_rate is not None and gt_cpr_rate is not None else None
                # Compute additional rates per minute if both are available
                gt_cpr_rate_per_min = gt_cpr_rate * 12 if gt_cpr_rate is not None else None
                pred_cpr_rate_per_min = pred_cpr_rate * 12 if pred_cpr_rate is not None else None
                anonymous_cpr_rate_per_min = anonymous_cpr_rate * 12 if anonymous_cpr_rate is not None else None
                anonymous_low_pass_cpr_rate_per_min = anonymous_low_pass_cpr_rate * 12 if anonymous_low_pass_cpr_rate is not None else None
                anonymous_low_pass_cpr_rate_per_min_error = anonymous_low_pass_cpr_rate_per_min - gt_cpr_rate_per_min if anonymous_low_pass_cpr_rate_per_min is not None and gt_cpr_rate_per_min is not None else None

                print("Low Pass anonymous Pred CPR rate: ", anonymous_low_pass_cpr_rate)

                # Append to data if participant and trial are found
                if participant and trial:
                    data.append({
                        'participant': participant,
                        'trial': trial,
                        'GT_Depth': gt_depth,
                        'Pred_Depth': pred_depth,
                        'Depth_error': depth_error,
                        'GT_CPR_rate': gt_cpr_rate,
                        'Pred_CPR_rate': pred_cpr_rate,
                        'CPR_rate_error': cpr_rate_error,
                        'GT_CPR_rate_per_min': gt_cpr_rate_per_min,
                        'Pred_CPR_rate_per_min': pred_cpr_rate_per_min,
                        'anonymous_Pred_CPR_rate': anonymous_cpr_rate,
                        'anonymous_Pred_CPR_rate_error': anonymous_cpr_rate_error,
                        'anonymous_Pred_CPR_rate_per_min': anonymous_cpr_rate_per_min,
                        'anonymous_Low_Pass_Pred_CPR_rate': anonymous_low_pass_cpr_rate,
                        'anonymous_Low_Pass_Pred_CPR_rate_error': anonymous_low_pass_cpr_rate_error,
                        'anonymous_Low_Pass_Pred_CPR_rate_per_min': anonymous_low_pass_cpr_rate_per_min,
                        'anonymous_Low_Pass_Pred_CPR_rate_per_min_error': anonymous_low_pass_cpr_rate_per_min_error

                    })
            except Exception as e:
                print(f"Error processing line: {line}\nError: {e}")

# Create DataFrame and calculate averages
df = pd.DataFrame(data)
# replace GT_Depth column from data from a different file
# Load original data and merge GT_Depth column based on participant
original_data = pd.read_csv('./original_averages_output.csv')
df = df.merge(original_data[['participant', 'GT_Depth']], on='participant', suffixes=('', '_from_original'))
# If there’s a column mismatch, make sure to replace the existing GT_Depth
df['GT_Depth'] = df['GT_Depth_from_original']
df = df.drop(columns=['GT_Depth_from_original'])


averages = df.groupby('participant').agg({
    'GT_Depth': 'mean',
    'Pred_Depth': 'mean',
    'Depth_error': 'mean',
    'GT_CPR_rate': 'mean',
    # 'Pred_CPR_rate': 'mean',
    # 'CPR_rate_error': 'mean',
    'GT_CPR_rate_per_min': 'mean',
    # 'Pred_CPR_rate_per_min': 'mean',
    # 'anonymous_Pred_CPR_rate': 'mean',
    # 'anonymous_Pred_CPR_rate_per_min': 'mean',
    # 'anonymous_Pred_CPR_rate_error': 'mean',
    # 'anonymous_Low_Pass_Pred_CPR_rate': 'mean',
    'anonymous_Low_Pass_Pred_CPR_rate_per_min': 'mean',
    # 'anonymous_Low_Pass_Pred_CPR_rate_error': 'mean',
    'anonymous_Low_Pass_Pred_CPR_rate_per_min_error': 'mean'
}).reset_index()


# Calculate squared error and mean squared error for each participant
averages['Depth_Error'] = (averages['GT_Depth'] - averages['Pred_Depth']) 

# Save averages to CSV
averages.to_csv(output_name, mode='w', index=False)

print("Averages have been saved to 'averages_output.csv'.")


# Calculate squared errors for RMSE
averages['Depth_Error_Squared'] = (averages['GT_Depth'] - averages['Pred_Depth']) ** 2
averages['anonymous_Low_Pass_Pred_CPR_rate_per_min_Error_Squared'] = averages['anonymous_Low_Pass_Pred_CPR_rate_per_min_error'] ** 2


# Calculate overall average across all subjects
overall_averages = averages.mean(numeric_only=True).to_frame().T
overall_averages.insert(0, 'participant', 'Overall')

# Calculate RMSE for Depth_error and anonymous_Low_Pass_Pred_CPR_rate_per_min_error
overall_averages['RMSE_Depth_Error'] = np.sqrt(averages['Depth_Error_Squared'].mean())
overall_averages['RMSE_anonymous_Low_Pass_Pred_CPR_rate_per_min_Error'] = np.sqrt(averages['anonymous_Low_Pass_Pred_CPR_rate_per_min_Error_Squared'].mean())

# Save overall averages to a new CSV
overall_output_name = 'overall_averages_output.csv'
overall_averages.to_csv(overall_output_name, mode='w', index=False)
print(f"Overall averages have been saved to '{overall_output_name}'.")

