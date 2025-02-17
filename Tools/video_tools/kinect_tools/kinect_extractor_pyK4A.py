from argparse import ArgumentParser
import cv2
import os
import numpy as np
from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback, Config
from pyk4a import ImageFormat

def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

def save_point_cloud(capture, output_path, index):
    point_cloud = capture.depth_point_cloud
    if point_cloud is not None:
        # Flatten the point cloud and save it
        file_path = os.path.join(output_path, f"point_cloud_{index}.ply")
        with open(file_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(point_cloud.size))
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for i in range(point_cloud.shape[0]):
                f.write("{} {} {}\n".format(point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2]))

def play(playback: PyK4APlayback, rgb_output_path: str, depth_output_path: str, ir_output_path: str, point_cloud_output_path: str):
    rgb_frame_index = 0
    depth_frame_index = 0
    ir_frame_index = 0
    point_cloud_index = 0

    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                rgb_frame_index += 1
                rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
                cv2.imshow("Color", rgb_image)
                cv2.imwrite(rgb_output_path + f"rgb_{rgb_frame_index}.png", rgb_image)
            if capture.depth is not None:
                depth_frame_index += 1
                depth_image = colorize(capture.depth, (None, 5000), colormap=cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_image)
                cv2.imwrite(depth_output_path + f"depth_{depth_frame_index}.png", depth_image)
            if capture.ir is not None:
                ir_frame_index += 1
                ir_image = colorize(capture.ir, (None, 500), colormap=cv2.COLORMAP_BONE)
                cv2.imshow("IR", ir_image)
                cv2.imwrite(ir_output_path + f"ir_{ir_frame_index}.png", ir_image)
            # if capture.depth is not None:
            #     point_cloud_index += 1
            #     save_point_cloud(capture, point_cloud_output_path, point_cloud_index)
                
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
    cv2.destroyAllWindows()

def main() -> None:
    filename: str = '../../../TestData/ng2_t0/ng2_t0_trimmed_final.mkv'
    output_path = "../../../TestData/ng2_t0/"
    rgb_output_path = output_path + "rgb/"
    depth_output_path = output_path + "depth/"
    ir_output_path = output_path + "ir/"
    point_cloud_output_path = output_path + "point_cloud/"

    os.makedirs(rgb_output_path, exist_ok=True)
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(ir_output_path, exist_ok=True)
    os.makedirs(point_cloud_output_path, exist_ok=True)

    print("Saving RGB frames to: ", rgb_output_path)
    print("Saving Depth frames to: ", depth_output_path)
    print("Saving IR frames to: ", ir_output_path)
    print("Saving Point Clouds to: ", point_cloud_output_path)

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)
    play(playback, rgb_output_path, depth_output_path, ir_output_path, point_cloud_output_path)

    playback.close()

if __name__ == "__main__":
    main()
