import os
import numpy as np
import matplotlib.pyplot as plt

data_path=r'D:\CPR_extracted\smartwatch_dataset'
log_path=r'Benchmarks/CPR_quality/GT/log_CPR_data.txt'

for sbj in range(0,24):
    part_data=np.load(os.path.join(data_path,'part_data.npy'))

    valid_args = np.argwhere(np.isin(part_data, sbj))
    gt_data=np.load(os.path.join(data_path,'gt_data.npy'))[valid_args,:].squeeze()
    peak_data=np.load(os.path.join(data_path,'gt_peak_data.npy'))[valid_args,:].squeeze()
    valley_data=np.load(os.path.join(data_path,'gt_valley_data.npy'))[valid_args,:].squeeze()

    depth_list,n_cpr_list=[],[]
    for i in range(gt_data.shape[0]):
        x=np.argwhere(peak_data[i,:])
        peak_vals=gt_data[i][x]
        x=np.argwhere(valley_data[i,:])
        valley_vals=gt_data[i][x]
        l=min(len(peak_vals),len(valley_vals))
        mean_depth=(peak_vals[:l]-valley_vals[:l]).mean().item()
        CPR_rate=(len(valley_vals)+len(peak_vals))*0.5/5*60
        if CPR_rate>190 or CPR_rate<40 or mean_depth>50 or mean_depth>70 or mean_depth<10:
            continue
        depth_list.append(mean_depth)
        n_cpr_list.append(CPR_rate)

    with open(log_path,'a') as f:
        f.write(f'subject {sbj}, mean depth: {np.mean(depth_list)}, std depth: {np.std(depth_list)}, CPR rate: {np.mean(n_cpr_list)}, std CPR rate: {np.std(n_cpr_list)} \n')
        

# plt.plot(gt_data[0,:])
# peak_x=np.argwhere(peak_data[0,:])
# peak_vals=gt_data[0][peak_x]
# plt.plot(peak_x,peak_vals,'ro')
# peak_x=np.argwhere(valley_data[0,:])
# peak_vals=gt_data[0][peak_x]
# plt.plot(peak_x,peak_vals,'ro')
# plt.show(block=True)







