import pandas as pd
import re
import os

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
                keshara_pred_cpr_rate_match = re.search(r"Keshara_Pred_CPR_rate:\[(\d+\.\d+)\]", line)
                keshara_low_pass_pred_cpr_rate_match = re.search(r"Low_Pass_Keshara_Pred_CPR_rate:\[(\d+\.\d+)\]", line)


                # Extracted data
                participant = participant_match.group(1) if participant_match else None
                trial = trial_match.group(1) if trial_match else None
                gt_depth = float(gt_depth_match.group(1)) if gt_depth_match else None
                pred_depth = float(pred_depth_match.group(1)) if pred_depth_match else None
                depth_error = float(depth_error_match.group(1)) if depth_error_match else None
                gt_cpr_rate = float(gt_cpr_rate_match.group(1)) if gt_cpr_rate_match else None
                pred_cpr_rate = float(pred_cpr_rate_match.group(1)) if pred_cpr_rate_match else None
                cpr_rate_error = float(cpr_rate_error_match.group(1)) if cpr_rate_error_match else None
                keshara_cpr_rate = float(keshara_pred_cpr_rate_match.group(1)) if keshara_pred_cpr_rate_match else None
                keshara_low_pass_cpr_rate = float(keshara_low_pass_pred_cpr_rate_match.group(1)) if keshara_low_pass_pred_cpr_rate_match else None
                keshara_cpr_rate_error = keshara_cpr_rate - gt_cpr_rate if keshara_cpr_rate is not None and gt_cpr_rate is not None else None
                keshara_low_pass_cpr_rate_error = keshara_low_pass_cpr_rate - gt_cpr_rate if keshara_low_pass_cpr_rate is not None and gt_cpr_rate is not None else None
                # Compute additional rates per minute if both are available
                gt_cpr_rate_per_min = gt_cpr_rate * 12 if gt_cpr_rate is not None else None
                pred_cpr_rate_per_min = pred_cpr_rate * 12 if pred_cpr_rate is not None else None
                keshara_cpr_rate_per_min = keshara_cpr_rate * 12 if keshara_cpr_rate is not None else None
                keshara_low_pass_cpr_rate_per_min = keshara_low_pass_cpr_rate * 12 if keshara_low_pass_cpr_rate is not None else None
                
                print("Low Pass Keshara Pred CPR rate: ", keshara_low_pass_cpr_rate)

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
                        'Keshara_Pred_CPR_rate': keshara_cpr_rate,
                        'Keshara_Pred_CPR_rate_error': keshara_cpr_rate_error,
                        'Keshara_Pred_CPR_rate_per_min': keshara_cpr_rate_per_min,
                        'Keshara_Low_Pass_Pred_CPR_rate': keshara_low_pass_cpr_rate,
                        'Keshara_Low_Pass_Pred_CPR_rate_error': keshara_low_pass_cpr_rate_error,
                        'Keshara_Low_Pass_Pred_CPR_rate_per_min': keshara_low_pass_cpr_rate_per_min,

                    })
            except Exception as e:
                print(f"Error processing line: {line}\nError: {e}")

# Create DataFrame and calculate averages
df = pd.DataFrame(data)
averages = df.groupby('participant').agg({
    'GT_Depth': 'mean',
    'Pred_Depth': 'mean',
    'Depth_error': 'mean',
    'GT_CPR_rate': 'mean',
    'Pred_CPR_rate': 'mean',
    'CPR_rate_error': 'mean',
    'GT_CPR_rate_per_min': 'mean',
    'Pred_CPR_rate_per_min': 'mean',
    'Keshara_Pred_CPR_rate': 'mean',
    'Keshara_Pred_CPR_rate_per_min': 'mean',
    'Keshara_Pred_CPR_rate_error': 'mean',
    'Keshara_Low_Pass_Pred_CPR_rate': 'mean',
    'Keshara_Low_Pass_Pred_CPR_rate_per_min': 'mean',
    'Keshara_Low_Pass_Pred_CPR_rate_error': 'mean'
}).reset_index()

# Save averages to CSV with header only on first write
file_exists = os.path.isfile(output_name)
averages.to_csv(output_name, mode='a', index=False, header=not file_exists)

print("Averages have been saved to 'averages_output.csv'.")
