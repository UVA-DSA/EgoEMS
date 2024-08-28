#import from models folder transtcn
from models.transtcn import TransformerModel
import torch

def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(args)
    model.to(device)
    return model
