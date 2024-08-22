import torch
import numpy as np
import json
from config import *
from utils import reset_parameters, traintest_loop, rolling_average
from models import initiate_model
from utils import json_to_csv

import datetime
import argparse

torch.manual_seed(0)

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
model_name = args.model
# verbose_mode = args.verbose

torch.manual_seed(0)

model,optimizer,scheduler,criterion = initiate_model(input_dim=input_dim,output_dim=output_dim,transformer_params=transformer_params,learning_params=learning_params, tcn_model_params=tcn_model_params, model_name=model_name)

print(model)
