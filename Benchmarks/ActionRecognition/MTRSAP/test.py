from utils.utils import *
from scripts.config import DefaultArgsNamespace
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = DefaultArgsNamespace()

model = init_model(args)

dummy_input = torch.randn(5, 30, 3, 224, 224) # batch, num_frames, channels, height, width

# feature extraction

dummy_input = dummy_input.to(args.device)

dummy_output = model(dummy_input)   

print(dummy_output.shape)