import torch
from torch import nn

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        # padding chosen so output length == input length:
        pad = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # if in_ch != out_ch, project x for the skip
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) \
                          if in_ch != out_ch else None

    def forward(self, x):
        # x: [B, in_ch, T]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # [B, out_ch, T] matched

class TCNDepthNet(nn.Module):
    def __init__(self, in_channels=3, channels=(64,128,256),
                 kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            prev = in_channels if i == 0 else channels[i-1]
            layers.append(ResidualBlock(prev, ch,
                                        kernel_size,
                                        dilation=2**i,
                                        dropout=dropout))
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # → [B, channels[-1], 1]
            nn.Flatten(),             # → [B, channels[-1]]
            nn.Dropout(0.2),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)          # → [B,1]
        )

    def forward(self, x):
        # x: [B, T=150, C=3]
        x = x.permute(0, 2, 1)       # → [B, 3, 150]
        y = self.tcn(x)             # → [B, 256, 150]
        out = self.head(y)          # → [B,1]
        return out.squeeze(-1)      # → [B]

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
    