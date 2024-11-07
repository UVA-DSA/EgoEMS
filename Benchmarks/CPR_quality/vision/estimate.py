import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
parent_directory = os.path.abspath('.')
print(parent_directory)
sys.path.append(parent_directory)
from Tools.depth_sensor_processing import tools as depth_tools
import sys
import cv2
import extract_depth
import torch
import re

sample_rate = 30



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

GT_path=r'D:\EgoExoEMS_CVPR2025\Dataset\Final'
data_path=r'D:\Final\exo_kinect_cpr_clips'
log_path=r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\vision_log.txt'
debug_plots_path=r'E:\EgoExoEMS\Benchmarks\CPR_quality\vision\debug_plots'
os.makedirs(debug_plots_path, exist_ok=True)


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

        
        # commented for now. not needed for rate detection. needed for depth detection
        #read depth images
        # rgb_imgs,depth_imgs=extract_depth.read_video(os.path.join(data_dir,mkv_file))

        # if not len(rgb_imgs)==len(json_data.keys()):
        #     continue

        keys=[int(k) for k in json_data.keys()]
        keys.sort()
        wrist_x,wrist_y=[],[]
        key_list=[]

                
        for k in keys:
            kpt_data = json_data.get(str(k), {})
            hands = kpt_data.get('hands', {})
            # print("frame: ",k, " hands:", hands)
            if len(hands)==0:
                continue
            wrist_x.append(hands[0]['x'][0])
            wrist_y.append(hands[0]['y'][0])
            key_list.append(k)


        print(f'Processing {json_file} and mkv {mkv_file} with {len(keys)} frames')
        print(f'Found  {len(wrist_x)} wrist x keypoints, {len(wrist_y)} wrist y keypoints')

                # Use regular expression to find two floating point numbers in the filename
        match = re.search(r"(\d+\.\d+)_(\d+\.\d+)_exo\.mkv", mkv_file)

        start_t = None
        end_t = None
        start_frame = None
        end_frame = None
        if match:
            start_t = float(match.group(1))
            end_t = float(match.group(2))
            start_frame = int(start_t * sample_rate)
            end_frame = int(end_t * sample_rate)
            print("Start time:", start_t)
            print("End time:", end_t)

            print("Start frame:", start_frame)
            print("End frame:", end_frame)
        else:
            print("Pattern not found in filename.")



        wrist_y=np.array(wrist_y,dtype=float)
        wrist_x=np.array(wrist_x,dtype=float)

        # plt.title(f'Wrist X keypoints for {json_file.split(".")[0]}')
        # plt.plot(wrist_x)
        # plt.show(block=True)


        plt.title(f'Wrist Y keypoints for {json_file.split(".")[0]}')
        plt.plot(wrist_y)
        plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_wrist_y.png')
        # close the plot to continue
        plt.close()
        # plt.show(block=True)

        # diff_y=np.abs(np.diff(wrist_y))
        # diff_x=np.abs(np.diff(wrist_x))
        # outlier_idx=np.where(diff_y>10)[0]
        # wrist_y[outlier_idx]=np.nan
        # outlier_idx=np.where(diff_x>10)[0]
        # wrist_x[outlier_idx]=np.nan

        # #filter out outliers
        # vals=wrist_y**2
        # vals=(vals-vals.min())/(vals.max()-vals.min())
        # outlier_idx=np.where(vals>0.3)[0]
        # wrist_y[outlier_idx]=np.nan

        # vals=wrist_x**2
        # vals=(vals-vals.min())/(vals.max()-vals.min())
        # outlier_idx=np.where(vals>0.3)[0]
        # wrist_x[outlier_idx]=np.nan

        # #interpolate
        # not_nan = np.where(~np.isnan(wrist_y))[0]
        # wrist_y[outlier_idx] = np.interp(outlier_idx, not_nan, wrist_y[not_nan])
        # not_nan = np.where(~np.isnan(wrist_x))[0]
        # wrist_x[outlier_idx] = np.interp(outlier_idx, not_nan, wrist_x[not_nan])

        # plt.plot(wrist_y)
        # plt.show(block=True)

        #detect peaks and valleys
        # apply low pass
        tensor_wrist_y = torch.tensor(wrist_y)
        try:
            filtered_wrist_y = depth_tools.low_pass_filter(tensor_wrist_y, 30)
        except Exception as e:
            print(f"Error applying low pass filter: {e}")
            continue
        plt.title(f'Filtered Wrist Y keypoints for {json_file.split(".")[0]}')
        plt.plot(filtered_wrist_y)
        plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_filtered_wrist_y.png')
        plt.close()

        # plt.show(block=True)


        p,v=depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(filtered_wrist_y),mul=1,show=False)
        # p,v=depth_tools.detect_peaks_and_valleys_depth_sensor(np.array(wrist_y),mul=1,show=False)
        n_cpr=(len(p)+len(v))*0.5
        print(f'Number of Predicted CPR cycles for the clip: {n_cpr}')

        #get GT cpr quality
        sbj=json_file.split('_')[0]
        t=json_file.split('_')[1][1:]

        gt_path=os.path.join(GT_path,sbj,'cardiac_arrest',t,'distance_sensor_data','sync_depth_sensor.csv')
        with open(gt_path, 'r') as file:
            gt_lines = file.readlines()

        gt_lines = np.array([int(round(float(line.strip()))) for line in gt_lines[1:]])
        gt_readings_for_clip=gt_lines[start_frame:end_frame]

        print("number of depth sensor readings: ", len(gt_lines))
        print("number of depth sensor readings for clip: ", len(gt_readings_for_clip))

        # plot the depth sensor readings for the clip and save
        plt.plot(gt_readings_for_clip)
        plt.title(f'Depth Sensor Readings for {json_file.split(".")[0]}')
        plt.savefig(f'{debug_plots_path}/{json_file.split(".")[0]}_depth_sensor_readings.png')
        plt.close()



        gt_peaks,gt_valleys=depth_tools.detect_peaks_and_valleys_depth_sensor(gt_readings_for_clip,mul=1,show=False)
        gt_n_cpr=(len(gt_peaks)+len(gt_valleys))*0.5
        print(f'Number of GT CPR cycles for the clip: {gt_n_cpr}')

        log_msg = f"Subject: {sbj}, {mkv_file.split('.')[0]} GT CPR cycles: {gt_n_cpr}, Predicted CPR cycles: {n_cpr}"
        write_log_line(log_path, log_msg)
        # peak_depths=gt_lines[gt_peaks]
        # valley_depths=gt_lines[gt_valleys]
        # l=min(len(peak_depths),len(valley_depths))
        # gt_cpr_depth=float((peak_depths[:l]-valley_depths[:l]).mean())



    #     break
    # break

        # depth_imgs=np.array(depth_imgs)
        # depth_imgs_p=depth_imgs[p]
        # depth_imgs_v=depth_imgs[v]
        # wrist_x_p=wrist_x[p]
        # wrist_y_p=wrist_y[p]
        # wrist_x_v=wrist_x[v]
        # wrist_y_v=wrist_y[v]

        # peakXYZ_p=[]
        # for idx in range(len(depth_imgs_p)):
        #     depth_img=depth_imgs_p[idx]
        #     d=int(depth_img[int(wrist_y_p[idx]),int(wrist_x_p[idx])])
        #     X,Y,Z=get_XYZ(int(wrist_x_p[idx]),int(wrist_y_p[idx]),d,CANON_K)
        #     peakXYZ_p.append([float(X),float(Y),float(Z)])
        # peakXYZ_v=[]
        # for idx in range(len(depth_imgs_v)):
        #     depth_img=depth_imgs_v[idx]
        #     d=int(depth_img[int(wrist_y_v[idx]),int(wrist_x_v[idx])])
        #     X,Y,Z=get_XYZ(int(wrist_x_v[idx]),int(wrist_y_v[idx]),d,CANON_K)
        #     peakXYZ_v.append([float(X),float(Y),float(Z)])

        # l=min(len(peakXYZ_p),len(peakXYZ_v))
        
        # dist_list=[]
        # for idx in range(l):
        #     p_=np.array(peakXYZ_p[idx])
        #     v_=np.array(peakXYZ_v[idx])
        #     dist=np.sum((p_-v_)**2)**0.5
        #     dist_list.append(float(dist))
        # cpr_depth=float(np.mean(dist_list))

        # #get GT cpr quality
        # sbj=json_file.split('_')[0]
        # t=json_file.split('_')[1][1:]

        # gt_path=os.path.join(GT_path,sbj,'cardiac_arrest',t,'distance_sensor_data','sync_depth_sensor.csv')
        # with open(gt_path, 'r') as file:
        #     gt_lines = file.readlines()
        # gt_lines = np.array([int(line.strip()) for line in gt_lines[1:]])
        # gt_peaks,gt_valleys=depth_tools.detect_peaks_and_valleys_depth_sensor(gt_lines,mul=5,show=False)
        # gt_n_cpr=(len(gt_peaks)+len(gt_valleys))*0.5
        # peak_depths=gt_lines[gt_peaks]
        # valley_depths=gt_lines[gt_valleys]
        # l=min(len(peak_depths),len(valley_depths))
        # gt_cpr_depth=float((peak_depths[:l]-valley_depths[:l]).mean())

        # time=len(gt_lines)/30

        # depth_err=abs(gt_cpr_depth-cpr_depth)
        # n_cpr_err=abs(gt_n_cpr-n_cpr)

        # msg=f'Subject: {sbj} CPR depth error : {depth_err} mm, CPR frequency error : {n_cpr_err/time*60} /min'
        # print(msg)
        # write_log_line(log_path,msg)


        # plt.imshow(rgb_imgs[idx])
        # plt.scatter(int(wrist_x_p[idx]), int(wrist_y_p[idx]), color='red')
        # plt.title(f'Depth Image at Peak {idx}')
        # plt.show(block=True)




