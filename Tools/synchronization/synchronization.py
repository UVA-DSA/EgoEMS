import os, sys
import pandas


goPro_timestamps = "path to folder of generated timestamps from gopro_telemetry.js"
offsets = "path to csv of offsets, with two columns: day, offset, where offset is gopro time - computer time"

goPro_json = pandas.DataFrame['filename', 'start_frame', 'end_frame']
dS_json = pandas.DataFrame['filename', 'start_frame', 'end_frame']

def synchronize(day):
    for person in os.walk(day):
        for intervention in os.walk(person):
            for trial in os.walk(intervention):
                for modality in os.walk(trial):  
                    for file in os.walk(modality):
                        if file.startswith("VL"): #finding the depth sensor data csv
                            depth_data = pandas.read_csv(os.path.join(day, person, intervention, trial, modality, file))
                        if file.startswith("sw_data"): #finding the smartwatch data csv
                            sw_data = pandas.read_csv(os.path.join(day, person, intervention, trial, modality, file))
                        if file.endswith(".mkv"): #finding the depth camera video, save the path to send to trimmer
                            dc_video_path = os.path.join(day, person, intervention, trial, modality, file)
                        if file.endswith(".txt"): #finding the depth camera time stamp text file
                            with open(os.path.join(day, person, intervention, trial, modality, file), 'r') as depth_txt:
                                dc_ts = depth_txt.readlines() #get a list of depth camera time stamps
                        if file.endswith(".mp4"): #finding the gopro video
                            gp_video_path = os.path.join(day, person, intervention, trial, modality, file)
                            for csv in os.walk(goPro_timestamps):
                                if os.path.splitext(file)[0] == os.path.splittext(csv)[0]: #match go pro timestamp file to go pro video name
                                    gp_ts = pandas.read_csv(csv)
                                    for row in offsets:
                                        if offsets['day'][row] == str(day):
                                            gp_adj_ts = gp_ts['epoch'] - offsets['offset'][row]
                                        else:
                                            print(f"couldn't match {day} to a row in offset csv")
                                else:
                                    print(f"issue with finding go pro timestamp csv matching {gp_video_path} for {day}_{person}_{intervention}_{trial}") 
                        
                        
                        #find starting frames for go pro and depth camera
                if dc_ts is not None & gp_adj_ts is not None:
                    if dc_ts[0] > gp_adj_ts[0]: #case where depth camera starts later than go pro
                        dc_sf = 0
                        dc_length = len(dc_ts)
                        for i in range(len(gp_adj_ts) - 1): 
                            if gp_adj_ts[i] < dc_ts[0] and gp_adj_ts[i + 1] > dc_ts[0]: #find the first timestamp where go pro ts is earlier than depth cam, and following go pro ts is later than depth cam
                                gp_sf = i
                                gp_length = len(gp_adj_ts)- i
                                break
                    else: #case where go pro starts later than depth camera
                        gp_sf = 0
                        gp_length = len(gp_adj_ts)
                        for i in range(len(dc_ts) - 1): 
                            if dc_ts[i] < gp_adj_ts[0] and dc_ts[i + 1] > gp_adj_ts[0]: #find the first timestamp where go pro ts is earlier than depth cam, and following go pro ts is later than depth cam
                                dc_sf = i
                                dc_length = len(gp_adj_ts)- i
                                break
                else:   
                    print(f"issue with either go pro adjusted timestamps or depth cam timestamps")
                if dc_sf is None or gp_sf is None:
                    print(f"issue with finding go pro starting frame or depth camera start frame")
                        
                #find ending frames for go pro and depth camera
                if gp_length > dc_length: #first case where gopro (after evening out the beginnings) is longer than the depth camera 
                    frames = dc_length #find the number of synchronized frames for the trial
                else:
                    frames = gp_length
                    
                gp_ef = gp_sf + frames
                dc_ef = dc_sf + frames

                if ((gp_ef - gp_sf) != (dc_ef - dc_sf)):
                    print(f"issue with equal number of frames for depth camera ({dc_sf} - {dc_ef}) and go pro ({gp_sf} - {gp_ef})")

                #collect starting and ending frames in dataframes to be converted to jsons for clipping
                goPro_json.append({'filename' : gp_video_path, 'start_frame' : gp_sf, 'end_frame': gp_ef})
                dS_json.append({'filename' : dc_video_path, 'start_frame' : dc_sf, 'end_frame': dc_ef})


                #synchronize smartwatch and depth sensor data
                if sw_data is None or sw_data.empty:
                    print(f"WARNING: sw file empty for {person} {intervention} {trial}")
                    continue
                if depth_data is None or depth_data.empty:
                    print(f"WARNING: depth sensor data empty for {person} {intervention} {trial} ")
                    continue

                
                sw_timestamps = sw_data['sw_epoch_ms'].astype('int64')
                ds_timestamps = depth_data[depth_data.columns[0]].astype('int64')

                sw_index = 0
                depth_index = 0

                #get timestamps 
                timestamps = gp_adj_ts[gp_sf:gp_ef]
                
                # Lists to store synchronized data
                sw_data_selected_list = []
                depth_data_selected_list = []

                
            # Synchronize data based on timestamps
            for i, gopro_epoch in enumerate(timestamps):
                found_sw_data = False
                while sw_index < len(sw_timestamps) - 1:
                    if gopro_epoch >= sw_timestamps[sw_index] * 1000000 and gopro_epoch < sw_timestamps[sw_index + 1] * 1000000:
                        sw_data_selected_list.append({
                            'sw_value_X_Axis': sw_data['value_X_Axis'].iloc[sw_index],
                            'sw_value_Y_Axis': sw_data['value_Y_Axis'].iloc[sw_index],
                            'sw_value_Z_Axis': sw_data['value_Z_Axis'].iloc[sw_index],
                        })
                        found_sw_data = True
                        sw_index += 1
                        break
                #for this timestamp, if we cannot find sw data, we should put in zeroes
                if not found_sw_data:
                    sw_data_selected_list.append({
                        'sw_value_X_Axis': 0,
                        'sw_value_Y_Axis': 0,
                        'sw_value_Z_Axis': 0,
                    })

                found_ds_data = False
                while depth_index < len(ds_timestamps) - 1:
                    if gopro_epoch >= ds_timestamps[depth_index] and gopro_epoch < ds_timestamps[depth_index + 1]:
                        depth_data_selected_list.append({
                            'depth_value': depth_data[depth_data.columns[1]].iloc[depth_index],
                        })
                        found_ds_data = True
                        depth_index += 1
                        break
                #for this timestamp, if we cannot find ds data, we should put in zeroes
                if not found_ds_data:
                    depth_data_selected_list.append({
                            'depth_value': 0,
                        }) 
            

                # Create DataFrames from the selected data
                df_sw = pandas.DataFrame(sw_data_selected_list)
                df_ds = pandas.DataFrame(depth_data_selected_list)

                # Write smartwatch data to CSV
                sw_out = os.path.join("synchronized", day, person, intervention, trial, "smartwatch_data")
                os.makedirs(sw_out, exist_ok=True)
                sw_filename = f"{day}_{person}_{intervention}_{trial}_smartwatch.csv"
                df_sw.to_csv(os.path.join(sw_out, sw_filename), index=False)

                # Write depth sensor data to CSV
                ds_out = os.path.join("synchronized", day, person, intervention, trial, "depthSensor_data")
                ds_filename = f"{day}_{person}_{intervention}_{trial}_depthSensor.csv"
                df_ds.to_csv(os.path.join(ds_out, ds_filename), index=False)





#this function takes in which day we want to synchronize
#in the bash file synchronize.sh, there should be a variable
#"folders" which contains the names of all the folders in the 
#directory that contains all synchronization python scripts that
#are wanted to be synchronized

#this function should have access to a csv called synchronization, 
#which lists the day, and gopro-computer time offset

#this function should output 2 json files (one for gopro, one for kinect) of starting and ending frames for 
#each go pro and kinect file to clip, and an organized directory of synchronized smartwatch and depth sensor data
if __name__ == "__main__":
    if len(sys.argv) > 1:
        day = sys.argv[1]
        synchronize(day)
    else: 
        print("no folder provided")