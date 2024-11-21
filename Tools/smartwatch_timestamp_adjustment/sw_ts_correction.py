import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('/standard/storage/EgoExoEMS_CVPR2025/Dataset/anonymous/P4/s4/smartwatch_data/sw_data.csv')

# Drop rows where server_epoch_ms is missing to calculate the offset
data_with_both_epochs = data.dropna(subset=['server_epoch_ms', 'sw_epoch_ms'])

# Calculate the offset between server_epoch_ms and sw_epoch_ms
offset = int(np.mean(data_with_both_epochs['server_epoch_ms'] - data_with_both_epochs['sw_epoch_ms']))

# Fill in missing server_epoch_ms based on sw_epoch_ms and the calculated offset
data['filled_server_epoch_ms'] = data['server_epoch_ms'].fillna(data['sw_epoch_ms'] + offset).round().astype(int)

# Save or use the modified data as needed
data.to_csv('./filled_epochs.csv', index=False)

print(data.head())
