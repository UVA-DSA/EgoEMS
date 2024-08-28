from utils.utils import *
from scripts.config import DefaultArgsNamespace

args = DefaultArgsNamespace()

model = init_model(args)

dummy_input = torch.randn(5, 30, 64)
dummy_input = dummy_input.to(args.device)

dummy_output = model(dummy_input)   

print(dummy_output.shape)