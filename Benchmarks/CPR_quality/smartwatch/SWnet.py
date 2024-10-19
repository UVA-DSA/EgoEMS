import torch
from torch import nn

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