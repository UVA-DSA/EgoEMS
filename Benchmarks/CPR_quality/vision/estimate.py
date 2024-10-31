import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from Tools.depth_sensor_processing import tools as depth_tools
import sys
import cv2
import extract_depth


CANON_K=np.array([[615.873673811006,0,640.803032851225],[0,615.918359977960,365.547839233105],[0,0,1]])

def get_XYZ(x,y,depth,k):
    X=(x-k[0,2])*depth/k[0,0]
    Y=(y-k[1,2])*depth/k[1,1]
    Z=depth
    return X,Y,Z

def init_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
        # create the log folder if it does not exist
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

def write_log_line(log_path, msg):
    with open(log_path, 'a') as file:
        file.write(msg+'\n')

GT_path=r'C:\Users\lahir\Downloads\data\standard\UVA-DSA\NIST EMS Project Data\EgoExoEMS_CVPR2025\Dataset\Final'
data_path=r'C:\Users\lahir\Downloads\data\standard\UVA-DSA\NIST EMS Project Data\EgoExoEMS_CVPR2025\Dataset\Kinect_CPR_Clips\exo_kinect_cpr_clips'
log_path=r'Benchmarks/CPR_quality/vision/log.txt'

init_log(log_path)

for n in ['train_root','test_root','val_root']:
    data_dir=os.path.join(data_path,n,'chest_compressions')

    json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
    mkv_files=[file for file in os.listdir(data_dir) if file.endswith('.mkv')]


    def detect_outliers_mad(data, threshold=3):
        median = np.median(data)
        mad = np.median([abs(x - median) for x in data]) + 1e-6
        if np.any(mad == 0):
            raise ValueError("Data contains zero values, which may lead to division by zero.")
        return [i for i,x in enumerate(data) if abs(x - median) / mad > threshold]

    for json_file in json_files:
        json_path=os.path.join(data_dir,json_file)
        json_data = json.load(open(json_path))
        mkv_file=[f for f in mkv_files if json_file.replace('_keypoints.json','') in f][0]

        #read depth images
        rgb_imgs,depth_imgs=extract_depth.read_video(os.path.join(data_dir,mkv_file))

        if not len(rgb_imgs)==len(json_data.keys()):
            continue

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
        wrist_x=np.array(wrist_x,dtype=float)

        # plt.plot(wrist_x)
        # plt.show(block=True)

        # diff_y=np.abs(np.diff(wrist_y))
        # diff_x=np.abs(np.diff(wrist_x))
        # outlier_idx=np.where(diff_y>10)[0]
        # wrist_y[outlier_idx]=np.nan
        # outlier_idx=np.where(diff_x>10)[0]
        # wrist_x[outlier_idx]=np.nan

        #filter out outliers
        vals=wrist_y**2
        vals=(vals-vals.min())/(vals.max()-vals.min())
        outlier_idx=np.where(vals>0.3)[0]
        wrist_y[outlier_idx]=np.nan

        vals=wrist_x**2
        vals=(vals-vals.min())/(vals.max()-vals.min())
        outlier_idx=np.where(vals>0.3)[0]
        wrist_x[outlier_idx]=np.nan

        #interpolate
        not_nan = np.where(~np.isnan(wrist_y))[0]
        wrist_y[outlier_idx] = np.interp(outlier_idx, not_nan, wrist_y[not_nan])
        not_nan = np.where(~np.isnan(wrist_x))[0]
        wrist_x[outlier_idx] = np.interp(outlier_idx, not_nan, wrist_x[not_nan])

        # plt.plot(wrist_y)
        # plt.show(block=True)

        #detect peaks and valleys
        p,v=depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(wrist_y),mul=1,show=False)
        n_cpr=(len(p)+len(v))*0.5

        depth_imgs=np.array(depth_imgs)
        depth_imgs_p=depth_imgs[p]
        depth_imgs_v=depth_imgs[v]
        wrist_x_p=wrist_x[p]
        wrist_y_p=wrist_y[p]
        wrist_x_v=wrist_x[v]
        wrist_y_v=wrist_y[v]

        peakXYZ_p=[]
        for idx in range(len(depth_imgs_p)):
            depth_img=depth_imgs_p[idx]
            d=int(depth_img[int(wrist_y_p[idx]),int(wrist_x_p[idx])])
            X,Y,Z=get_XYZ(int(wrist_x_p[idx]),int(wrist_y_p[idx]),d,CANON_K)
            peakXYZ_p.append([float(X),float(Y),float(Z)])
        peakXYZ_v=[]
        for idx in range(len(depth_imgs_v)):
            depth_img=depth_imgs_v[idx]
            d=int(depth_img[int(wrist_y_v[idx]),int(wrist_x_v[idx])])
            X,Y,Z=get_XYZ(int(wrist_x_v[idx]),int(wrist_y_v[idx]),d,CANON_K)
            peakXYZ_v.append([float(X),float(Y),float(Z)])

        l=min(len(peakXYZ_p),len(peakXYZ_v))
        
        dist_list=[]
        for idx in range(l):
            p_=np.array(peakXYZ_p[idx])
            v_=np.array(peakXYZ_v[idx])
            dist=np.sum((p_-v_)**2)**0.5
            dist_list.append(float(dist))
        cpr_depth=float(np.mean(dist_list))

        #get GT cpr quality
        sbj=json_file.split('_')[0]
        t=json_file.split('_')[1][1:]

        gt_path=os.path.join(GT_path,sbj,'cardiac_arrest',t,'distance_sensor_data','sync_depth_sensor.csv')
        with open(gt_path, 'r') as file:
            gt_lines = file.readlines()
        gt_lines = np.array([int(line.strip()) for line in gt_lines[1:]])
        gt_peaks,gt_valleys=depth_tools.detect_peaks_and_valleys_depth_sensor(gt_lines,mul=5,show=False)
        gt_n_cpr=(len(gt_peaks)+len(gt_valleys))*0.5
        peak_depths=gt_lines[gt_peaks]
        valley_depths=gt_lines[gt_valleys]
        l=min(len(peak_depths),len(valley_depths))
        gt_cpr_depth=float((peak_depths[:l]-valley_depths[:l]).mean())

        time=len(gt_lines)/30

        depth_err=abs(gt_cpr_depth-cpr_depth)
        n_cpr_err=abs(gt_n_cpr-n_cpr)

        msg=f'Subject: {sbj} CPR depth error : {depth_err} mm, CPR frequency error : {n_cpr_err/time*60} /min'
        print(msg)
        write_log_line(log_path,msg)




        # plt.imshow(rgb_imgs[idx])
        # plt.scatter(int(wrist_x_p[idx]), int(wrist_y_p[idx]), color='red')
        # plt.title(f'Depth Image at Peak {idx}')
        # plt.show(block=True)




