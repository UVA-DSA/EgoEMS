import pandas as pd

raw_data_path = "/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw"
goPro_timestamps = f"{raw_data_path}/goPro_timestamps/GX010321.csv"

df = pd.read_csv(goPro_timestamps)

# Get the first epoch time
first_epoch_time = df['epoch'].iloc[0]

# Calculate the recalculated epoch time by adding CTS to the first epoch time
# CTS is in seconds, convert it to nanoseconds and add it to the first epoch time
# in integer format
df['recalculated_epoch'] = (df['cts'] * 1e6).astype(int) + first_epoch_time




df.to_csv(goPro_timestamps, index=False)