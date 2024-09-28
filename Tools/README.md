## Useful commands

### FFMPEG extract RGB stream from Kinect Recording
```bash 
ffmpeg -i 2024-09-05-19-07-43_trimmed.mkv -map 0:v:0 -c:v libx264 -crf 23 -preset fast rgb_stream.mp4 
```

### FFMPEG combine GoPro and Kinect RGB view 
```bash
ffmpeg -i GoPro/GX010334_trimmed.mp4 -i Kinect/rgb_stream.mp4 \
-filter_complex "[0:v]scale=1920:1080,fps=29.97,setpts=PTS-STARTPTS[v0]; \
[1:v]scale=1920:1080,fps=29.97,setpts=PTS-STARTPTS[v1]; \
[v0][v1]hstack=inputs=2,format=yuv420p[v]" \
-map "[v]" -map 0:a? -shortest SynchedPreview/output.mp4
```

### MKVMerge Trim
```python
command = [
   'mkvmerge',
   '-o', output_file,  # Output file
   '--split', f'parts:{start_time_formatted}-{end_time_formatted}',  # Split using timestamps
   filepath]
```

### MKVMerge Resample
```python
command = [
    'mkvmerge',
    '-o', output_file,  # Output file
    '--default-duration', f'0:{target_fps}fps',  # Split using timestamps
    filepath
]
```

## Synchronization Steps

### Setup

- **Root Directory:** - `/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/`
- **Day:** - `"05-09-2024"`

## Sync Steps

### Steps

1. **Re-Encode Gopro**
   Run the gopro_reencoder.sh by folder.
   
3. **Convert Frame Rate of Kinect to GoPro Frame Rate**
   Use kinect_fps_convert.sh to downsample kinect recordings.
   
5. **File Path Extract**
   Run the file path extractor. (This will run for the whole september directory, given that the goPro files have been reencoded)

6. **Visual Synch**
   Use dual_video_viewer.py to synchronize the depth camera and gopro video. This will output offset csvs in the synch_offset folder.

   Manually remove synched videos from file path list.

7. **Synch Main**
   Update synch_main.sh "sync_offset_dir" with the synchronized offset folder. Then ./sync_main.sh.

**The following is achieved by running sync_main.sh:**
1. **Adjust GoPro Timestamp Offset**  
   Run the following command to adjust the GoPro's timestamp using the offset between GoPro time and Kinect time (PC time):
   ```bash
   python goPro_timestamp_adjuster.py root_dir
   ```

2. **Generate Synchronization Metadata**  
   Generate synchronization metadata by running the script:
   ```bash
   python synchronization-v2.py root_dir day
   ```

3. **Convert Kinect Frame Rate to 29.97 FPS**  
   Adjust the Kinect recording frame rate to 29.97 FPS for consistency with the GoPro:
   ```bash
   python kinect_fps_converter.py root_dir
   ```

4. **Trim GoPro Recordings**  
   Use the synchronization metadata to trim the GoPro recordings:
   ```bash
   python gopro_trimmer.py root_dir
   ```

5. **Trim Kinect Recordings**  
   Similarly, trim the Kinect recordings using the synchronization metadata:
   ```bash
   python kinect_trimmer.py root_dir
   ```

6. **Create Side-by-Side Preview**  
   Create a side-by-side preview of synchronized GoPro and Kinect videos:
   ```bash
   python sync_clip_merger.py root_dir day

