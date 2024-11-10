# my data
# find /mnt/standard/NIST\ EMS\ Project\ Data/EgoExoEMS_CVPR2025/Dataset/Final -type f -name "*trimmed_final.mkv" -exec rsync -R {} /media/kesharaw/MIDAS_1 \;
# lahirus data
 find /mnt/standard/NIST\ EMS\ Project\ Data/EgoExoEMS_CVPR2025/Dataset/Final -type f   -name "*sync_depth_sensor.csv" -exec rsync -R {} /mnt/d/ \;
