import numpy as np
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def get_data_stats(data_loader):
    import torch
    depth_norm_vals=torch.empty(0)
    acc_vals=torch.empty(0)

    for i, batch in enumerate(data_loader):
        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0
        depth_gt_min = torch.where(depth_gt_mask, depth_gt, torch.tensor(float('inf')))
        mins=depth_gt_min.min(dim=1).values
        depth_norm=depth_gt-mins.unsqueeze(1)
        depth_norm_vals=torch.concat([depth_norm_vals,depth_norm[depth_gt_mask]])
        acc_vals=torch.concat([acc_vals,data])

    print(f'max depth: {depth_norm_vals.max()}')
    print(f'min depth: {depth_norm_vals.min()}')
    print(f'max acc: {torch.max(torch.max(acc_vals,dim=0).values,dim=0).values}')
    print(f'min acc: {torch.min(torch.min(acc_vals,dim=0).values,dim=0).values}')


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


def plot_line(data):
    plt.plot(data)
    plt.show(block=True)


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

