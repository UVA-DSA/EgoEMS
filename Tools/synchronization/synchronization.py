import os, sys
import pandas, numpy


raw_data_path = "/standard/storage/CognitiveEMS_Datasets/anonymous/Sep_2024/Raw"

goPro_timestamps = f"{raw_data_path}/goPro_timestamps"
offsets = pandas.read_csv(f"{raw_data_path}/offsets.csv")


def synchronize(day):
    goPro_json = pandas.DataFrame(columns=['filename', 'start_frame', 'end_frame'])
    dS_json = pandas.DataFrame(columns=['filename', 'start_frame', 'end_frame'])

    for person in os.listdir(day):
        person_path = os.path.join(day, person)
        if person != "berend":
            continue
        for intervention in os.listdir(person_path):
            intervention_path = os.path.join(person_path, intervention)
            for trial in os.listdir(intervention_path):
                if trial != "1":
                    continue
                trial_path = os.path.join(intervention_path, trial)
                sw_data = None
                depth_data = None
                gp_ts = None
                dc_video_path = None
                dc_ts = None
                gp_video_path = None
                gp_adj_ts = None
                for modality in os.listdir(trial_path):
                    modality_path = os.path.join(trial_path, modality)  
                        
                    for file in os.listdir(modality_path):
                        #print(f"searching for files in {modality_path}")
                        
                        if file.startswith("VL"): #finding the depth sensor data csv
                            depth_data = pandas.read_csv(os.path.join(modality_path, file))
                            print(f"found DEPTH SENSOR data")
                        if file.startswith("sw"): #finding the smartwatch folder
                            sw_path = os.path.join(modality_path, file)
                            print(f"searching for sw data file in {sw_path}")
                            for sw in os.listdir(sw_path):
                                sw_data = pandas.read_csv(os.path.join(sw_path, sw))
                                print(f"set SMARTWATCH data")
                        if file.endswith(".mkv"): #finding the depth camera video, save the path to send to trimmer
                            dc_video_path = os.path.join(modality_path, file)
                            print(f"set DEPTH CAMERA VIDEO path to {dc_video_path}")
                        if file.endswith(".txt"): #finding the depth camera time stamp text file
                            with open(os.path.join(modality_path, file), 'r') as depth_txt:
                                txt_list = depth_txt.readlines()#get a list of depth camera time stamps
                                dc_ts = numpy.array([int(line.strip()) for line in txt_list], dtype=numpy.int64)
                                print(f"found DEPTH CAMERA TS data!")
                        if file.endswith(".MP4"): #finding the gopro video
                            gp_video_path = os.path.join(modality_path, file)
                            print(f"found GO PRO VIDEO")
                            for csv in os.listdir(goPro_timestamps):
                                #print(f"searching csv {csv}")
                                if os.path.splitext(file)[0] == os.path.splitext(csv)[0]: #match go pro timestamp file to go pro video name
                                    print(f"identified {csv} as a match")
                                    gp_path = os.path.join(goPro_timestamps, csv)
                                    gp_ts = pandas.read_csv(gp_path)
                                    print(f"found file to match to {csv}")
                                    matching_offset = offsets[offsets['day'] == day]
                                    if not matching_offset.empty:
                                        offset_value = matching_offset.iloc[0, 1]
                                        print(f"we've identified an offset: {offset_value}")
                                        gp_adj_ts = gp_ts['epoch'].astype('int64') - offset_value
                                        print(f"{gp_adj_ts}")
                                    else:
                                            print(f"couldn't match {day} to a row in offset csv")
                            if gp_ts is None:
                                print(f"issue with finding go pro timestamp csv matching {gp_video_path} for {day}_{person}_{intervention}_{trial}") 
                        
                    
                #find starting frames for go pro and depth camera
                #print(f"this is dc_ts: {dc_ts[0]} {type(dc_ts[0])} and this is gp_adj_ts: {gp_adj_ts[0]} {type(gp_adj_ts[0])}")
                
                try:
                    if len(dc_ts) > 0 and len(gp_adj_ts) > 0 :
                        print(f"comparing dc_ts[0] : {dc_ts[0]} {type(dc_ts[0])} and gp_adj_ts[0]: {gp_adj_ts[0]} {type(gp_adj_ts[0])}")
                        if numpy.int64(dc_ts[0]) > numpy.int64(gp_adj_ts[0]): #case where depth camera starts later than go pro
                            print("decided depth camera stats later than go pro")
                            dc_sf = 0
                            dc_length = len(dc_ts)
                            for i in range(len(gp_adj_ts) - 1): 
                                if numpy.int64(gp_adj_ts[i]) <= numpy.int64(dc_ts[0]) and numpy.int64(gp_adj_ts[i + 1]) >= numpy.int64(dc_ts[0]): #find the first timestamp where go pro ts is earlier than depth cam, and following go pro ts is later than depth cam
                                    print(f"decided that {gp_adj_ts[i]} < {dc_ts[0]} < {gp_adj_ts[i + 1]}")
                                    print(f"setting gp_sf to {i}, a with gp_length {len(gp_adj_ts) - i}")
                                    gp_sf = i
                                    gp_length = len(gp_adj_ts)- i
                                    break
                        else: #case where go pro starts later than depth camera
                            print("decided depth camera starts earlier than go pro")
                            gp_sf = 0
                            gp_length = len(gp_adj_ts)
                            for i in range(len(dc_ts) - 1): 
                                if numpy.int64(dc_ts[i]) <= numpy.int64(gp_adj_ts[0]) and numpy.int64(dc_ts[i + 1]) >= numpy.int64(gp_adj_ts[0]): #find the first timestamp where go pro ts is earlier than depth cam, and following go pro ts is later than depth cam
                                    print(f"decided that {dc_ts[i]} < {gp_adj_ts[0]} < {dc_ts[i + 1]}")
                                    print(f"setting dc_sf to {i}, a with dc_length {len(dc_ts) - i}")
                                    dc_sf = i
                                    dc_length = len(dc_ts)- i
                                    break
                        
                                
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
                        new_goPro = {'filename' : gp_video_path, 'start_frame' : gp_sf, 'end_frame': gp_ef}
                        new_dS = {'filename' : dc_video_path, 'start_frame' : dc_sf, 'end_frame': dc_ef}
                        
                        new_goPro_df = pandas.DataFrame([new_goPro])
                        new_dS_df = pandas.DataFrame([new_dS])
                        goPro_json = pandas.concat([goPro_json, new_goPro_df], ignore_index=True)
                        dS_json = pandas.concat([dS_json, new_dS_df], ignore_index=True)
                        print(f"added {new_goPro} to goPro_json {goPro_json}")
                        print(f"added {new_dS} to ds_json {dS_json}")

                except:
                    print("ISSUE WITH GO PRO OR DEPTH CAMERA TIMESTAMPS")

                try:
                    if len(sw_data) > 0  and len(depth_data) > 0:
                    #synchronize smartwatch and depth sensor data
                        
                        sw_timestamps = sw_data['sw_epoch_ms'].astype('int64')
                        ds_timestamps = depth_data[depth_data.columns[0]].astype('int64')
                        print(f"ds_timestamps: {ds_timestamps}")
                        

                        sw_index = 0
                        depth_index = 0

                        #get timestamps 
                        timestamps = gp_adj_ts[gp_sf:gp_ef]
                        
                        # Lists to store synchronized data
                        sw_data_selected_list = []
                        depth_data_selected_list = []

                        
                        # Synchronize data based on timestamps
                        for i, gopro_epoch in enumerate(timestamps):
                            #print(f"gopro_epoch: {gopro_epoch}")
                            found_sw_data = False
                            while sw_index < len(sw_timestamps) - 1:
                                if gopro_epoch >= sw_timestamps[sw_index] * 1000000 and gopro_epoch < sw_timestamps[sw_index + 1] * 1000000:
                                    sw_data_selected_list.append({
                                        'sw_value_X_Axis': sw_data['value_X_Axis'].iloc[sw_index],
                                        'sw_value_Y_Axis': sw_data['value_Y_Axis'].iloc[sw_index],
                                        'sw_value_Z_Axis': sw_data['value_Z_Axis'].iloc[sw_index],
                                    })
                                    #print("sw data found!")
                                    found_sw_data = True
                                    break
                                sw_index += 1
                                    
                            #for this timestamp, if we cannot find sw data, we should put in zeroes
                            if not found_sw_data:
                                #print("MISSING SW DATA")
                                sw_data_selected_list.append({
                                    'sw_value_X_Axis': 0,
                                    'sw_value_Y_Axis': 0,
                                    'sw_value_Z_Axis': 0,
                                })
                                sw_index = 0

                            found_ds_data = False
                            #print(f"this is gopro_epoch: {gopro_epoch} {type(gopro_epoch)} and this is ds_timestamp: {ds_timestamps[depth_index]} {type(ds_timestamps[depth_index])}")
                            while depth_index < len(ds_timestamps) - 1:
                                #print(depth_index)
                                if numpy.int64(gopro_epoch) >= ds_timestamps[depth_index] and numpy.int64(gopro_epoch) < ds_timestamps[depth_index + 1]:
                                    depth_data_selected_list.append({
                                        'depth_value': depth_data[depth_data.columns[1]].iloc[depth_index],
                                    })
                                    #print("depth camera data found!")
                                    found_ds_data = True
                                    break
                                depth_index += 1
                                    
                            #for this timestamp, if we cannot find ds data, we should put in zeroes
                            if not found_ds_data:
                                #print("missing DEPTH DATA")
                                depth_data_selected_list.append({
                                        'depth_value': 0,
                                    }) 
                                depth_index = 0
                except:
                    print("ISSUE WITH SMARTWATCH OR DEPTH DATA")

                # Create DataFrames from the selected data
                try:
                    df_sw = pandas.DataFrame(sw_data_selected_list)
                    df_ds = pandas.DataFrame(depth_data_selected_list)

                    print(f"this is df_ds: {df_ds}")

                    # Write smartwatch data to CSV
                    sw_out = os.path.join("synchronized", day, person, intervention, trial, "smartwatch_data")
                    os.makedirs(sw_out, exist_ok=True)
                    sw_filename = f"{day}_{person}_{intervention}_{trial}_smartwatch.csv"
                    df_sw.to_csv(os.path.join(sw_out, sw_filename), index=False)

                    # Write depth sensor data to CSV
                    ds_out = os.path.join("synchronized", day, person, intervention, trial, "depthSensor_data")
                    ds_filename = f"{day}_{person}_{intervention}_{trial}_depthSensor.csv"
                    os.makedirs(ds_out, exist_ok=True)
                    df_ds.to_csv(os.path.join(ds_out, ds_filename), index=False)

                    print("creating CSVs of smartwatch and depth sensor data")
                except:
                    print("ISSUE WITH CREATING CSVs OF SMARTWATCH AND DEPTH SENSOR DATA")
                    
                    

    goPro_json.to_json("goPro_clip.json")
    dS_json.to_json("depthCam_clip.json")





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
        day = os.path.join(raw_data_path, day)
        synchronize(day)
    else: 
        print("no folder provided")