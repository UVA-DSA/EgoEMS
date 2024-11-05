import open3d as o3d
import numpy as np
import cv2

# Define paths to save RGB, Depth, and Point Cloud videos
mkv_path = "../../../TestData/wa1_t2_trimmed_final.mkv"
rgb_video_path = "../../../TestData/rgb_output_2.mp4"
depth_video_path = "../../../TestData/depth_output_2.mp4"
pcd_video_path = "../../../TestData/pointcloud_output_2.mp4"

# Define flags to enable or disable each stream
enable_rgb_stream = True
enable_depth_stream = True
enable_pointcloud_stream = True

# Create a reader for the MKV file
reader = o3d.io.AzureKinectMKVReader()

try:
    # Open the MKV file
    reader.open(mkv_path)

    # Check the first frame to get dimensions and initialize video writers if enabled
    frame = reader.next_frame()
    if frame is None:
        raise ValueError("No frames found in MKV file.")

    # Initialize variables for width and height
    width, height = None, None

    if enable_rgb_stream and frame.color is not None:
        rgb_np = np.asarray(frame.color)
        height, width, _ = rgb_np.shape

    # Get the camera's intrinsic parameters (replace with actual values if known)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=600,  # replace with actual focal length in x-axis
        fy=600,  # replace with actual focal length in y-axis
        cx= width / 2,
        cy= height / 2,
    )

    # Initialize RGB video writer if enabled
    if enable_rgb_stream and frame.color is not None:
        rgb_np = np.asarray(frame.color)
        height, width, _ = rgb_np.shape
        rgb_writer = cv2.VideoWriter(rgb_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    # Initialize Depth video writer if enabled
    if enable_depth_stream and frame.depth is not None:
        depth_np = np.asarray(frame.depth)
        depth_height, depth_width = depth_np.shape
        depth_writer = cv2.VideoWriter(depth_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (depth_width, depth_height))

    # Initialize Point Cloud video writer and visualizer if enabled
    if enable_pointcloud_stream and width is not None and height is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        pointcloud_writer = cv2.VideoWriter(pcd_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    # Process frames from the recording
    while not reader.is_eof():
        frame = reader.next_frame()
        if frame is None:
            continue

        # Get RGB and depth images if enabled
        rgb_image = frame.color if enable_rgb_stream else None
        depth_image = frame.depth if enable_depth_stream else None

        # Write RGB frame to video if enabled
        if enable_rgb_stream and rgb_image is not None:
            rgb_np = np.asarray(rgb_image)
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            rgb_writer.write(rgb_np)
            cv2.imshow("RGB Image", rgb_np)

        # Write Depth frame to video if enabled
        if enable_depth_stream and depth_image is not None:
            depth_np = np.asarray(depth_image)
            depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = np.uint8(depth_norm)
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            depth_writer.write(depth_colormap)
            cv2.imshow("Depth Heatmap", depth_colormap)

        # Render and record the point cloud if enabled
        if enable_pointcloud_stream and rgb_image is not None and depth_image is not None:
            # Create RGBD image from color and depth images
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb_image)),
                o3d.geometry.Image(np.asarray(depth_image)),
                convert_rgb_to_intensity=False
            )

            # Generate point cloud from RGBD image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

            # Render point cloud to the visualizer and capture it
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Capture the screen image of the point cloud
            pcd_img = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
            pcd_img = pcd_img.astype(np.uint8)
            pointcloud_writer.write(pcd_img)
            cv2.imshow("Point Cloud", pcd_img)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    print("Finished reading the file and saving videos.")

except Exception as e:
    print(f"Error opening or processing the MKV file: {e}")

finally:
    # Close the reader and video writers if they were initialized
    reader.close()
    if enable_rgb_stream and 'rgb_writer' in locals():
        rgb_writer.release()
    if enable_depth_stream and 'depth_writer' in locals():
        depth_writer.release()
    if enable_pointcloud_stream and 'pointcloud_writer' in locals():
        pointcloud_writer.release()
    if enable_pointcloud_stream and 'vis' in locals():
        vis.destroy_window()

# Clean up OpenCV windows
cv2.destroyAllWindows()
