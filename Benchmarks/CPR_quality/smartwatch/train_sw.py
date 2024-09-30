import hydra
from dataloader import sw_dataloader as dataloader
from model.SWnet import SWNET
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import utils
import numpy as np
import os
import pandas as pd

criterion = nn.MSELoss()

def eval(model,data_loader,conf,k=30,height=0.15,show_plots=False):
    depth_min,depth_max=conf.smartwatch.depth_min,conf.smartwatch.depth_max
    n_comp_min,n_comp_max=conf.smartwatch.n_comp_min,conf.smartwatch.n_comp_max
    signal_error_mean,depth_error_mean,mean_comp_error=0,0,0
    gt_compressions_list,depth_error_list,gt_depth_list,comp_error_list=[],[],[],[]

    for batch in data_loader:
        sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
        pred_signal,pred_depth=model(sw_data)
        pred_depth=pred_depth*(depth_max-depth_min)+depth_min
        gt_depth=gt_depth*(depth_max-depth_min)+depth_min

        # plt.plot(gt[0,:].detach().numpy())
        # plt.plot(pred_signal.detach().numpy())
        #detect peaks
        pred=pred_signal.detach().numpy()
        pred=(pred-min(pred))/(max(pred)-min(pred))
        pred=pred-np.mean(pred)
        t=conf.smartwatch.window_len
        num_zero_crossings = len(np.where(np.diff(np.sign(pred)))[0])/t
        dist=int(1/num_zero_crossings*k)
        # plt.plot(pred)
        pred_peaks, pred_valleys,_=utils.find_peaks_and_valleys(pred,distance=dist,height=height,plot=False)
        n_compressions=0.5*(len(pred_peaks)+len(pred_valleys))
        gt_compressions=0.5*(len(peaks[peaks==1])+len(valleys[valleys==1]))

        comp_error=(n_compressions-gt_compressions)**2
        mean_comp_error+=comp_error

        signal_loss=criterion(pred_signal,gt)
        depth_loss=torch.square(pred_depth-gt_depth).item()
        signal_error_mean+=signal_loss.item()
        depth_error_mean+=depth_loss

        gt_compressions_list.append(gt_compressions)
        depth_error_list.append(depth_loss)
        gt_depth_list.append(gt_depth.item())
        comp_error_list.append(comp_error)


        
    signal_error_mean/=len(data_loader)
    depth_error_mean/=len(data_loader)
    mean_comp_error/=len(data_loader)
    
    print('------------------------------------')
    print(f'Test : Mean signal error: {signal_error_mean} | RMSE depth : {depth_error_mean**0.5:.4f} mm | RMSE n_comp_error (per minute): {mean_comp_error**0.5*12:.4f}')
    print('------------------------------------')

    if show_plots:

        CPR_rate=conf.smartwatch.eval_settings.CPR_rate
        CPR_depth=conf.smartwatch.eval_settings.CPR_depth
        save_path=conf.smartwatch.eval_settings.plot_save_path

        gt_depth_list=np.array(gt_depth_list)
        depth_error_list=np.array(depth_error_list)
        comp_error_list=np.array(comp_error_list)
        gt_compressions_list=np.array(gt_compressions_list)
        
        plt.figure()
        plt.hist(gt_depth_list, bins=20)
        plt.xlabel('GT depth (mm)')
        plt.ylabel('Number of samples')
        d=conf.smartwatch.eval_settings.CPR_depth
        plt.axvline(x=0.5*(d[0]+d[1]), color='red', linestyle='--')
        plt.text(0.5, 0.7, f'Total Samples : {len(gt_depth_list)}', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(save_path,'depth_hist.png'), dpi=500)


        plt.figure()
        plt.hist(gt_compressions_list*12, bins=20)
        plt.xlabel('Number of compressions per minute')
        r=conf.smartwatch.eval_settings.CPR_rate
        plt.axvline(x=0.5*(r[0]+r[1]), color='red', linestyle='--')
        plt.ylabel('Number of samples')
        plt.text(0.5, 0.7, f'Total Samples : {len(gt_depth_list)}', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(save_path,'compression_hist.png'), dpi=500)


        #plot depth error vs gt depth
        depth_sort_idx=np.argsort(gt_depth_list)
        gt_depth=gt_depth_list[depth_sort_idx]
        d_errors=depth_error_list[depth_sort_idx]
        c_errors=comp_error_list[depth_sort_idx]

        good_depth_idx=np.where((gt_depth<=CPR_depth[-1]) & (gt_depth>=CPR_depth[0]))
        good_mean_depth_error=np.mean(d_errors[good_depth_idx])
        good_mean_comp_error=np.mean(c_errors[good_depth_idx])

        gt_depth=gt_depth.astype(int)
        df=pd.DataFrame()
        df['d']=gt_depth
        df['d_error']=d_errors
        df['c_error']=c_errors
        df_grouped = df.groupby('d').mean()
        df_grouped['d_error']=df_grouped['d_error']**0.5
        df_grouped['c_error']=df_grouped['c_error']**0.5*12

        
        plt.figure()
        plt.scatter(df_grouped.index, df_grouped['d_error'], s=30)
        y_min, y_max = plt.ylim()
        d=conf.smartwatch.eval_settings.CPR_depth
        plt.axvline(x=0.5*(d[0]+d[1]), color='red', linestyle='--')
        # plt.fill_between(df_grouped.index, y_max, alpha=0.2, where=(df_grouped.index >= CPR_depth[0]) & (df_grouped.index <= CPR_depth[-1]),color='limegreen')
        plt.xlabel('GT mean compression depth (mm)')
        plt.ylabel('Predicted depth error (mm)')
        # plt.text(0.7, 0.7, f'Mean error : {good_mean_depth_error:.1f}', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(save_path,'d_error_v_d.png'), dpi=500)


        plt.figure()
        plt.scatter(df_grouped.index, df_grouped['c_error'])
        y_min, y_max = plt.ylim()
        d=conf.smartwatch.eval_settings.CPR_depth
        plt.axvline(x=0.5*(d[0]+d[1]), color='red', linestyle='--')
        # plt.fill_between(df_grouped.index, y_max, alpha=0.2, where=(df_grouped.index >= CPR_depth[0]) & (df_grouped.index <= CPR_depth[-1]),color='limegreen')
        # plt.text(0.7, 0.7, f'Mean error : {good_mean_comp_error*12:.1f}', transform=plt.gca().transAxes)
        plt.xlabel('GT mean compression depth (mm)')
        plt.ylabel('Number of compression error per minute')
        plt.savefig(os.path.join(save_path,'c_error_v_d.png'), dpi=500)


        c_sort_idx=np.argsort(gt_compressions_list)
        c_list=gt_compressions_list[c_sort_idx]
        d_errors=depth_error_list[c_sort_idx]
        c_errors=comp_error_list[c_sort_idx]

        good_c_idx=np.where((gt_compressions_list*12<=CPR_rate[-1]) & (gt_compressions_list*12>=CPR_rate[0]))
        good_mean_depth_error=np.mean(d_errors[good_c_idx])
        good_mean_comp_error=np.mean(c_errors[good_c_idx])

        df=pd.DataFrame()
        df['c']=c_list
        df['d_error']=d_errors
        df['c_error']=c_errors
        df_grouped = df.groupby('c').mean()
        df_grouped['d_error']=df_grouped['d_error']**0.5
        df_grouped['c_error']=df_grouped['c_error']**0.5

        plt.figure()
        plt.scatter(df_grouped.index*12, df_grouped['d_error'])
        y_min, y_max = plt.ylim()
        # plt.fill_between(df_grouped.index*12, y_max, alpha=0.2, where=(df_grouped.index*12 >= CPR_rate[0]) & (df_grouped.index*12 <= CPR_rate[-1]),color='limegreen')
        # plt.text(0.32, 0.7, f'Mean error : {good_mean_depth_error:.1f}', transform=plt.gca().transAxes)
        # r=conf.smartwatch.eval_settings.CPR_rate
        plt.axvline(x=0.5*(r[0]+r[1]), color='red', linestyle='--')
        plt.xlabel('Number of compressions per minute')
        plt.ylabel('Depth error (mm)')
        plt.savefig(os.path.join(save_path,'d_error_vs_c.png'), dpi=500)

        plt.figure()
        plt.scatter(df_grouped.index*12, df_grouped['c_error']*12)
        y_min, y_max = plt.ylim()
        # plt.fill_between(df_grouped.index*12, y_max, alpha=0.2, where=(df_grouped.index*12 >= CPR_rate[0]) & (df_grouped.index*12 <= CPR_rate[-1]),color='limegreen')
        # plt.text(0.32, 0.7, f'Mean error : {good_mean_comp_error*12:.1f}', transform=plt.gca().transAxes)
        r=conf.smartwatch.eval_settings.CPR_rate
        plt.axvline(x=0.5*(r[0]+r[1]), color='red', linestyle='--')
        plt.xlabel('Number of compressions per minute')
        plt.ylabel('Number of compression error per minute')
        plt.savefig(os.path.join(save_path,'c_error_vs_c.png'), dpi=500)
    

def train(conf):

    model=SWNET(conf)
    model.train()
    model=model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.smartwatch.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    #create directory to save model
    if not os.path.exists(conf.smartwatch.model_save_path):
        os.makedirs(conf.smartwatch.model_save_path,exist_ok=True)

    train_dataloader, test_dataloader = dataloader.get_dataloaders(conf)
    for epoch in range(conf.smartwatch.epochs):
        mean_depth_loss,mean_signal_loss=0,0 
        for batch in train_dataloader:
            optimizer.zero_grad()
            sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
            pred_signal,pred_depth=model(sw_data)
            signal_loss=criterion(pred_signal,gt)
            depth_loss=criterion(pred_depth,gt_depth)
            # pred_valleys=criterion(pred_valleys,valleys)
            # depth_loss=criterion(pred_depth,gt_depth.squeeze())
            loss=signal_loss+depth_loss
            loss.backward()
            optimizer.step()
            
            mean_depth_loss+=depth_loss.item()
            mean_signal_loss+=signal_loss.item()
        mean_depth_loss/=len(train_dataloader)
        mean_signal_loss/=len(train_dataloader)
        print(f'Epoch: {epoch} Mean depth loss: {mean_depth_loss:.4f} Mean signal loss: {mean_signal_loss:.4f}')
        if (epoch+1)%conf.smartwatch.eval_freq==0:
            eval(model,test_dataloader,conf)
            torch.save(model.state_dict(), conf.smartwatch.model_save_path)
        # Step the scheduler
        scheduler.step()

def get_stats(conf):
    if conf.smartwatch.get_stats:
        dataloader.get_data_stats(conf)
   
def eval_checkpt(conf):
    model=SWNET(conf)
    model=model.double()
    model.load_state_dict(torch.load(conf.smartwatch.eval_settings.checkpoint_path))
    model.eval()

    _, test_dataloader = dataloader.get_dataloaders(conf)
    eval(model,test_dataloader,conf,k=conf.smartwatch.eval_settings.k,
         height=conf.smartwatch.eval_settings.height,
         show_plots=conf.smartwatch.eval_settings.show_plots)
    
def peak_detection(conf):
    _, test_dataloader = dataloader.get_dataloaders(conf)
    mean_error_list=[]
    gt_compressions_list,pred_compression_error_list=[],[]

    for j,batch in enumerate(test_dataloader):
        print(f'Batch: {j}/{len(test_dataloader)}',end='\r')
        sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
        i={
            'acc': 0,
            'gyr': 1,
            'mag': 2
        }  
        start_idx=i[conf.smartwatch.eval_settings.peak_detection_sensor]      
        sw_data=sw_data.squeeze().numpy()[start_idx:(start_idx+3)]
        prominant_axis=np.argmax(np.std(sw_data,axis=1))
        data=sw_data[prominant_axis,:]
        data_avg=(data-min(data))/(max(data)-min(data))
        data_avg=utils.moving_normalize(data,window_size=100)

        # plt.plot(data_avg)

        t=conf.smartwatch.window_len
        num_zero_crossings = len(np.where(np.diff(np.sign(data_avg)))[0])/t
        fit_window=int(400/num_zero_crossings)
        idx=np.arange(len(data_avg))/len(data_avg)
        data_int=utils.interpolate_between_ts(data_avg,idx,idx,fit_window=fit_window,deg=2)
        # plt.plot(data_int)
        # plt.plot(gt[0,:].numpy())
        # plt.show()
        # data_avg=utils.moving_normalize(data_int,window_size=300)

        # print(f'Zero crossings: {num_zero_crossings}')
        dist=int(1/num_zero_crossings*200)
        # data_int=data_int-np.mean(data_int)
        p, v,_=utils.find_peaks_and_valleys(data_int,distance=dist,height=0.0,plot=False)
        pred_n_comp=0.5*(len(p)+len(v))
        gt_compressions=0.5*(len(peaks[peaks==1])+len(valleys[valleys==1]))
        gt_compressions_list.append(gt_compressions)
        comp_error=(pred_n_comp-gt_compressions)**2
        pred_compression_error_list.append(comp_error)

    rmse_error=np.mean(pred_compression_error_list)**0.5*12
    print(f'RMSE num compression error per minute: {rmse_error}')

    if conf.smartwatch.eval_settings.show_plots:
        save_path=conf.smartwatch.eval_settings.plot_save_path
        gt_compressions_list=np.array(gt_compressions_list)
        pred_compression_error_list=np.array(pred_compression_error_list)
        idx=np.argsort(gt_compressions_list)
        gt_compressions_list=gt_compressions_list[idx]
        gt_compressions_list = gt_compressions_list.astype(int)
        pred_compression_error_list=pred_compression_error_list[idx]
        df=pd.DataFrame()
        df['gt']=gt_compressions_list
        df['error']=pred_compression_error_list
        df_grouped = df.groupby('gt').mean()
        gt=df_grouped.index
        error=df_grouped['error']**0.5
        plt.figure()
        plt.scatter(gt*12,error*12)
        plt.xlabel('GT number of compressions / minute')
        plt.ylabel('Predicted compression error (RMSE) / minute')
        plt.savefig(os.path.join(save_path,'compression_error_peak_detection.png'), dpi=500)


def FFT_est(conf):
    _, test_dataloader = dataloader.get_dataloaders(conf)
    mean_error_list=[]
    gt_compressions_list,pred_compression_error_list=[],[]

    for batch in test_dataloader:
        sw_data, gt_depth, gt,gt_n_comp,peaks,valleys=batch
        i={
            'acc': 0,
            'gyr': 1,
            'mag': 2
        }  
        start_idx=i[conf.smartwatch.eval_settings.peak_detection_sensor]      
        sw_data=sw_data.squeeze().numpy()[start_idx:(start_idx+3)]
        prominant_axis=np.argmax(np.std(sw_data,axis=1))
        data=sw_data[prominant_axis,:]
        dom_freq_est=utils.get_dominant_freq(data,conf.smartwatch.TARGET_FREQ)*60
        gt_compressions=(0.5*(len(peaks[peaks==1])+len(valleys[valleys==1])))/conf.smartwatch.window_len*60
        freq_error=(dom_freq_est-gt_compressions)**2
        mean_error_list.append(freq_error)
        gt_compressions_list.append(gt_compressions)

    mean_error=np.mean(mean_error_list)**0.5
    print(f'Mean num compression error (per minute): {mean_error}')
    df=pd.DataFrame()
    gt_compressions_list=np.array(gt_compressions_list).astype(int)
    df['gt']=gt_compressions_list
    df['error']=mean_error_list
    df_grouped = df.groupby('gt').mean()
    save_path=conf.smartwatch.eval_settings.plot_save_path
    plt.plot(df_grouped.index,df_grouped['error'],'o')
    plt.xlabel('GT number of compressions / minute')
    plt.ylabel('Predicted compression error / minute')
    plt.savefig(os.path.join(save_path,'compression_error_FFT.png'), dpi=500)



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(conf):
    if conf.smartwatch.mode=='train':
        train(conf)
    elif conf.smartwatch.mode=='eval':
        eval_checkpt(conf)
    elif conf.smartwatch.mode=='stats':
        get_stats(conf)
    elif conf.smartwatch.mode=='peak_detection':
        peak_detection(conf)
    elif conf.smartwatch.mode=='fft':
        FFT_est(conf)
        
if __name__ == "__main__":
    main()


'''
Fine - tuning

k	depth error	comp error height=0.15
20	8.07	0.88
30		0.76
40		0.64
50		0.58
60		0.55
70		0.54
80		0.53
90		0.56
100		0.66
		
		
k=80		
height	comp error	
0.001	0.4582	
0.01	0.4565	
0.05	0.4517	
0.1	0.4773	
0.15	0.53	


best k =80 height=0.05

'''