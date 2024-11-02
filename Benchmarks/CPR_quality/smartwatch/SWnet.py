import torch
from torch import nn
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerCPR(nn.Module):
    def __init__(self, in_channels=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerCPR, self).__init__()
        
        # Linear layer to project input (x, y, z) to model dimension
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling layer to summarize the sequence output
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer to predict CPR rate
        self.fc_rate = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: (batch_size, samples, in_channels)
        
        # Project the input to the Transformer model dimension
        x = self.input_projection(x)  # Shape: (batch_size, samples, d_model)
        
        # Permute for Transformer input format
        x = x.permute(1, 0, 2)  # Shape: (samples, batch_size, d_model)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x)  # Shape: (samples, batch_size, d_model)
        
        # Permute back to (batch_size, samples, d_model)
        transformer_out = transformer_out.permute(1, 2, 0)
        
        # Pool across the sequence dimension
        pooled_features = self.pooling(transformer_out).squeeze(-1)  # Shape: (batch_size, d_model)
        
        # Predict CPR rate
        pred_cpr_rate = self.fc_rate(pooled_features).squeeze(-1)  # Shape: (batch_size)
        
        return pred_cpr_rate




class SWNET(nn.Module):
    def __init__(self,in_channels=3,out_len=300):
        super(SWNET, self).__init__()
        self.out_len=out_len
        #********head 1*******
        self.l1_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )
        self.l1_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )
        
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc_signal = nn.Linear(64, 1)
        self.fc_depth = nn.Linear(64, 1)
        self.sm=nn.Softmax(dim=1)

        self.peak_detection = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )


    def forward(self, x):
        out1 = self.l1_1(x)
        out1 = self.l1_2(out1)
        out1 = out1.permute(0, 2, 1)

        lstm_out, (hn, cn) = self.lstm1(out1)
        bs,seq_len,hidden_dim=lstm_out.shape
        output=lstm_out.reshape(-1,hidden_dim)

        #reconstruct the original signal
        signal=self.fc_signal(output)
        signal=signal.reshape(bs,seq_len,1)
        signal=signal.swapaxes(1,2)
        signal_interpolated = torch.nn.functional.interpolate(signal, size=(self.out_len,), mode='linear').squeeze()

        #predict depth
        pred_depth=self.fc_depth(lstm_out[:,-1,:])
    
        return signal_interpolated,pred_depth.squeeze()
    


import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

def calculate_cpr_rate_batch(accel_data_batch, sampling_rate, window_duration=10):
    batch_size = accel_data_batch.shape[0]
    window_samples = int(window_duration * sampling_rate)

    # Low-pass filter function
    def lowpass_filter(data, cutoff=2.0, fs=50.0, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    # To store CPR rates for each batch
    cpr_rates = []

    for i in range(batch_size):
        # Compute the magnitude of acceleration for each sample in the batch
        accel_magnitude = np.sqrt(np.sum(accel_data_batch[i]**2, axis=1))

        # Apply low-pass filter
        accel_magnitude_filtered = lowpass_filter(accel_magnitude, cutoff=2.0, fs=sampling_rate)

        # Detect peaks in the filtered magnitude signal
        peaks, _ = find_peaks(accel_magnitude_filtered, distance=sampling_rate/2)  # Adjust peak distance as needed

        # Calculate the number of peaks within the specified window
        num_peaks_in_window = len([p for p in peaks if p < window_samples])

        # Calculate CPR rate in compressions per minute for this batch item
        cpr_rate = (num_peaks_in_window / window_duration) * 60
        cpr_rates.append(cpr_rate)

    # Convert results to a numpy array
    return np.array(cpr_rates)
