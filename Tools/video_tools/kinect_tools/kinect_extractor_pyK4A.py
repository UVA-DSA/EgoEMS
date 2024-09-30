from argparse import ArgumentParser

import cv2
import os

from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play(playback: PyK4APlayback, rgb_output_path: str, depth_output_path: str, ir_output_path: str):
    rgb_frame_index = 0
    depth_frame_index = 0
    ir_frame_index = 0
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
                depth_image = colorize(capture.depth, (None, 5000))
                cv2.imshow("Depth", depth_image)
                cv2.imwrite(depth_output_path + f"depth_{depth_frame_index}.png",depth_image)
            if capture.ir is not None:
                ir_frame_index += 1
                ir_image = colorize(capture.ir, (None, 500), colormap=cv2.COLORMAP_BONE)
                cv2.imshow("IR", ir_image)
                cv2.imwrite(ir_output_path + f"ir_{ir_frame_index}.png",ir_image)
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
    cv2.destroyAllWindows()


def main() -> None:

    filename: str = 'C:/Users/kesha/Downloads/2024-09-05-19-58-16_fps_converted_trimmed.mkv'

    output_path = "./outputs/debrah/cardiac_arrest/2/Kinect/"
    rgb_output_path = output_path + "rgb/"
    depth_output_path = output_path + "depth/"
    ir_output_path = output_path + "ir/"

    # make directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(rgb_output_path, exist_ok=True)
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(ir_output_path, exist_ok=True)

    print("Saving RGB frames to: ", rgb_output_path)
    print("Saving Depth frames to: ", depth_output_path)
    print("Saving IR frames to: ", ir_output_path)

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    play(playback, rgb_output_path, depth_output_path, ir_output_path)

    playback.close()


if __name__ == "__main__":
    main()