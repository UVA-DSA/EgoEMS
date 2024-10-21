import os
import sys
parent_directory = os.path.abspath('.')
sys.path.append(parent_directory)
from Dataset.pytorch_implementation.EgoExoEMS.EgoExoEMS.EgoExoEMS import EgoExoEMSDataset


data_path=r'C:\Users\lahir\data\DataForLahiru\DataForLahiru'
data = EgoExoEMSDataset(annotation_file=None,data_base_path=data_path,fps=0)

