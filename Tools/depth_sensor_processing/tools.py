import numpy as np
from scipy.signal import butter, filtfilt
import torch

def moving_normalize(signal, window_size):
    # Initialize the normalized signal with zeros
    normalized_signal = np.zeros(signal.shape)
    
    # Calculate the half window size for indexing
    half_window = window_size // 2
    
    for i in range(len(signal)):
        # Determine the start and end of the window
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(signal))
        
        # Calculate local mean and standard deviation
        local_mean = np.mean(signal[start:end])
        local_std = np.std(signal[start:end])
        
        # Normalize the current value
        if local_std > 0:  # Avoid division by zero
            normalized_signal[i] = (signal[i] - local_mean) / local_std
        else:
            normalized_signal[i] = signal[i] - local_mean
    
    return normalized_signal

def find_peaks_and_valleys(signal, distance=10,height=0.2,prominence=(None, None),plot=False):
    from scipy.signal import find_peaks
    peaks, p_properties  = find_peaks(signal, distance=distance,height=height,prominence=prominence)
    valleys, v_properties = find_peaks(-signal, distance=distance,height=height,prominence=prominence)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(signal)
        plt.scatter(peaks, signal[peaks], c='r', label='Peaks')
        plt.scatter(valleys, signal[valleys], c='g', label='Valleys')
        plt.legend()
        plt.show(block=True)
    best_idx=-1
    if len(peaks)>1:
        best_idx=np.argmax([p_properties['prominences'].mean(),v_properties['prominences'].mean()])
    return peaks, valleys,best_idx

def detect_peaks_and_valleys_depth_sensor(depth_vals,mul=1,show=True):
    depth_vals_norm_=moving_normalize(depth_vals, 19)
    num_zero_crossings = len(np.where(np.diff(np.sign(depth_vals_norm_)))[0])/len(depth_vals_norm_)
    dist=int(1/num_zero_crossings*mul)
    depth_vals_norm=moving_normalize(depth_vals, 60)
    GT_peaks,GT_valleys,idx=find_peaks_and_valleys(depth_vals_norm,distance=dist,height=0.2,plot=show)
    return GT_peaks,GT_valleys

'''
CLIP_LENGTH : length of clip in seconds
use: 
depth_gt=batch['depth_sensor'].squeeze()
is_CPR(depth_gt)
'''
def is_CPR(x,CLIP_LENGTH=5,std_thres=5,ratio_thres=0.5):
    import torch
    parts=x.split(CLIP_LENGTH)
    stds=[torch.std(p) for p in parts]
    n_bad=len([s for s in stds if s<std_thres])
    bad_ratio=n_bad/len(stds)
    print(bad_ratio)
    if bad_ratio>ratio_thres: return False
    else: return True



# low pass filter
def low_pass_filter(accel_magnitude, fs=30):

    # Define filter parameters
    cutoff = 4  # Cutoff frequency in Hz
    order = 4  # Filter order

    # Create Butterworth filter
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    smoothed_magnitude = torch.tensor(filtfilt(b, a, accel_magnitude.cpu().numpy()).copy())

    return smoothed_magnitude