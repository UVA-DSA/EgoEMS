import torch
import torch.nn as nn
from torchvision import models


class I3D_Backbone(nn.Module):
    def __init__(self):
        super(I3D_Backbone, self).__init__()
        # Load the pretrained I3D model
        # Assuming you have an I3D model implementation available
        self.backbone = models.video.r3d_18(pretrained=True)  # Example using a 3D ResNet as a substitute for I3D
        
        # Modify to suit I3D architecture if needed
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Keep up to the last conv block
        
        # Optional: Freeze I3D weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # AvgPool to get (batch_size, 512, 1, 1, 1)
        
    def forward(self, x):
        x = self.backbone(x)  # Extract features
        x = self.backbone_avgpool(x)  # Shape: (batch_size, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten the tensor: (batch_size, 512)
        return x