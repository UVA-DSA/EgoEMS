import open3d as o3d
import numpy as np
import cv2

def read_video(video_path):
    # Create a reader for the MKV file
    reader = o3d.io.AzureKinectMKVReader()
    print("Reading the MKV file: ", video_path)

    rgb_imgs, depth_imgs = [], []
    try:
        # Open the MKV file
        reader.open(video_path)

        # Start processing frames from the recording
        while not reader.is_eof():
            frame = reader.next_frame()
            if frame is None:
                continue

            # Get RGB and depth images
            rgb_image = frame.color
            depth_image = frame.depth

            if rgb_image is not None:
                rgb_np = np.asarray(rgb_image)
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                rgb_imgs.append(rgb_np.copy())  # Store a copy of each RGB frame

            if depth_image is not None:
                # Convert depth image to numpy array
                depth_np = np.asarray(depth_image)
                depth_imgs.append(depth_np.copy())  # Store a copy of each depth frame

        print("Finished reading the file.")
        
    except Exception as e:
        print(f"Error opening or processing the MKV file: {e}")

    finally:
        # Close the reader when done
        reader.close()

    return rgb_imgs, depth_imgs


# Uncomment the following lines to test the function with a file path
# path = r'C:\Users\lahir\Downloads\data\standard\anonymous\NIST EMS Project Data\EgoExoEMS_CVPR2025\Dataset\Kinect_CPR_Clips\exo_kinect_cpr_clips\train_root\chest_compressions\ng5_t6_ks4_13.565_30.455_exo.mkv'
# rgb_imgs, depth_imgs = read_video(path)
