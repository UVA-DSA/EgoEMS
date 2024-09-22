from pyk4a import Config, PyK4APlayback

# Path to your MKV file
mkv_file_path = "your_recorded_file.mkv"

# Open the MKV playback
playback = PyK4APlayback(mkv_file_path)
playback.open()

# Check if it has depth and color streams
print("Depth Mode:", playback.configuration.depth_mode)
print("Color Resolution:", playback.configuration.color_resolution)

# Read frames from the playback
while True:
    try:
        capture = playback.get_next_capture()
        if capture.depth is not None:
            # Process depth frame
            depth_image = capture.depth
            # Do something with depth_image, like saving or processing
            print("Depth Frame Shape:", depth_image.shape)
    except EOFError:
        print("End of recording")
        break

# Close the playback
playback.close()
