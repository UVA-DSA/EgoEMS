import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from Tools.depth_sensor_processing import tools as depth_tools
import sys

data_path=r'C:\Users\lahir\Downloads\data\standard\UVA-DSA\NIST EMS Project Data\EgoExoEMS_CVPR2025\Dataset\Kinect_CPR_Clips\exo_kinect_cpr_clips'

data_dir=os.path.join(data_path,'train_root','chest_compressions')

json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]


def detect_outliers_mad(data, threshold=3):
    median = np.median(data)
    mad = np.median([abs(x - median) for x in data]) + 1e-6
    if np.any(mad == 0):
        raise ValueError("Data contains zero values, which may lead to division by zero.")
    return [i for i,x in enumerate(data) if abs(x - median) / mad > threshold]

for json_file in json_files:
    json_path=os.path.join(data_dir,json_file)
    json_data = json.load(open(json_path))
    keys=[int(k) for k in json_data.keys()]
    keys.sort()
    wrist_x,wrist_y=[],[]
    key_list=[]
    for k in keys:
        kpt_data=json_data[str(k)]
        if len(kpt_data['x'])==0:
            continue
        wrist_x.append(json_data[str(k)]['x'][0])
        wrist_y.append(json_data[str(k)]['y'][0])
        key_list.append(k)
    wrist_y=np.array(wrist_y,dtype=float)

    #filter out outliers
    vals=wrist_y**2
    vals=(vals-vals.min())/(vals.max()-vals.min())
    outlier_idx=np.where(vals>0.3)[0]
    wrist_y[outlier_idx]=np.nan

    #interpolate
    not_nan = np.where(~np.isnan(wrist_y))[0]
    wrist_y[outlier_idx] = np.interp(outlier_idx, not_nan, wrist_y[not_nan])

    # plt.plot(wrist_y)
    # plt.show(block=True)

    #detect peaks and valleys
    p,v=depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(wrist_y),mul=2,show=True)

    n_cpr=(len(p)+len(v))*0.5

    



    plt.plot(wrist_y)
    plt.show(block=True)

