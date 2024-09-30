import open3d as o3d
import numpy as np
import cv2



# Define the path to your Kinect recording (MKV file)
mkv_path = "c:/Users/kesha/Downloads/2024-09-05-19-58-16.mkv"

# Create a reader for the MKV file
reader = o3d.io.AzureKinectMKVReader()

try:
    # Open the MKV file
    reader.open(mkv_path)

    # Start processing frames from the recording
    # Start processing frames from the recording
    while not reader.is_eof():
        frame = reader.next_frame()
        if frame is None:
            continue

        # Get RGB and depth images
        rgb_image = frame.color
        depth_image = frame.depth

        if rgb_image is not None:
            # Convert RGB image to numpy array
            rgb_np = np.asarray(rgb_image)
            # convert to BGR
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imshow("RGB Image", rgb_np)

        if depth_image is not None:
            # Convert depth image to numpy array
            depth_np = np.asarray(depth_image)

            # Normalize the depth data for better visualization
            depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)

            # Convert depth to uint8 to apply the color map
            depth_uint8 = np.uint8(depth_norm)

            # Apply heatmap (color map) to the depth image
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

            # Show depth heatmap
            cv2.imshow("Depth Heatmap", depth_colormap)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    print("Finished reading the file.")
    
except Exception as e:
    print(f"Error opening or processing the MKV file: {e}")

finally:
    # Close the reader when done
    reader.close()

# Clean up OpenCV windows
cv2.destroyAllWindows
