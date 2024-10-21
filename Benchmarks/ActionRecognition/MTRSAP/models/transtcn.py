import math

import torch
import torch.nn as nn

from scripts.config import DefaultArgsNamespace

import torchvision.models as models
import torchaudio.transforms as transforms

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class GlobalMaxPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        
        return torch.max(input, axis=self.step_axis).values
    
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, x): #(batch, feature, seq)
        divider = torch.max(torch.max(torch.abs(x), dim=0)[0], dim=1)[0] + 1e-5
        divider = divider.unsqueeze(0).unsqueeze(2)
        divider = divider.repeat(x.size(0), 1, x.size(2))
        x = x / divider
        return x    

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)

        return out
    
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
    
class CNN_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),  # No change in temporal dimension
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),  # No change in temporal dimension
            nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1)  # No change in temporal dimension
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x
    
    
class CNN_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=96, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose1d(in_channels=96, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose1d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.decoder(x)
        # print('decoder_out',x.shape)
        return x
    
    
class TransformerModel(nn.Module):
    def __init__(self, args: DefaultArgsNamespace):
        super().__init__()

        self.input_dim = args.transformer_params["input_dim"]
        self.output_dim = args.transformer_params["output_dim"]
        self.d_model = args.transformer_params["d_model"]
        self.dropout = args.transformer_params["dropout"]
        self.num_layers = args.transformer_params["num_layers"]
        self.nhead = args.transformer_params["nhead"]
        self.batch_first = args.transformer_params["batch_first"]

        self.modality = args.dataloader_params["modality"]

        self.encoder_params = args.tcn_model_params["encoder_params"]
        self.decoder_params = args.tcn_model_params["decoder_params"]

        self.encoder_params["in_channels"] = self.input_dim
        self.decoder_params["out_channels"] = self.output_dim

        # Load ResNet-50 backbone and remove the final classification layer
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Keep up to the last conv block
        self.backbone_avgpool = nn.AdaptiveAvgPool2d((1, 1))  # AvgPool to get (batch_size, 2048, 1, 1)
        # Freeze ResNet-50 weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dropout=self.dropout, batch_first=self.batch_first),
            num_layers=self.num_layers
        )
        
        self.encoder = CNN_Encoder(**self.encoder_params)
        self.decoder = CNN_Decoder(**self.decoder_params)
        
        self.max_pool = GlobalMaxPooling1D()
        self.out = nn.Linear(self.d_model, self.output_dim)
        
        features_dim = 2048  # Output dimension of ResNet-50 backbone
        self.pe = PositionalEncoding(d_model=self.d_model, max_len=32, dropout=self.dropout)
        self.fc = nn.Linear(features_dim, self.d_model)


        # MelSpectrogram transform
        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=args.transformer_params['sample_rate'],
            n_mels=args.transformer_params['n_mels'],
            hop_length=args.transformer_params['hop_length'],
            n_fft=args.transformer_params['n_fft']
        )

        # Linear layer to project from 64 to 256
        self.projection_layer = nn.Linear(args.transformer_params['n_mels'], args.transformer_params['input_dim'] )
        self.projection_layer_multimodal = nn.Linear(args.transformer_params['n_mels'], args.transformer_params['resnet_dim'] )

        
    def extract_resnet(self, x):
        # check the shape of the input tensor
            # extract resnet50 features
        batch_size, num_frames, c, h, w = x.size()

        x = x.float()  # Ensures input is a FloatTensor


        # Reshape to process each frame individually through ResNet-50
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features using backbone
        x = self.backbone(x)  # Shape: (batch_size * num_frames, 2048, 7, 7)
        x = self.backbone_avgpool(x)  # Shape: (batch_size * num_frames, 2048, 1, 1)
        x = x.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, 2048)
        
        # x = self.fc(x)
        output = x

        return output

    def extract_mel_spectrogram(self, x, multimodal=False):
        # Extract Mel-spectrogram for both channels
        mel_spec_left = self.mel_spectrogram(x[:, :, 0])  # Left channel
        mel_spec_right = self.mel_spectrogram(x[:, :, 1])  # Right channel

        mel_spec_combined = torch.cat((mel_spec_left, mel_spec_right), dim=-1)  # Shape: [n_mels, time*2]
        feature = mel_spec_combined.permute(0, 2, 1)  # Shape: [batch, time*2, n_mels]
        if multimodal:
            feature = self.projection_layer_multimodal(feature)
        else:
            feature = self.projection_layer(feature)
        return feature


    def forward(self, x):
        
        # # TCN encoder
        x = self.encoder(x)
        # print("encoder_out",x.shape)
        x = x.permute(0, 2, 1)
        
        # # Add positional encoding
        x = self.pe(x)
        # print("pe_out",x.shape)
        
        # # Transformer expects input of shape (batch_size, seq_len, d_model)
        x = self.transformer(x)

        # # Further processing can be done here (e.g., pooling, classification)
        x = self.out(x)
        x = self.max_pool(x)  # Shape: (batch_size, output_dim)
        
        return x
    
