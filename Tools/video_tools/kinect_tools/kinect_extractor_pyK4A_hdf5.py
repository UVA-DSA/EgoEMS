import os
import h5py
import cv2
import numpy as np
from pyk4a import PyK4APlayback
from helpers import colorize

def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

def play(playback: PyK4APlayback, output_hdf5_path: str):
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        depth_group = hdf5_file.create_group("depth")
        ir_group = hdf5_file.create_group("ir")

        depth_frame_index = 0
        ir_frame_index = 0

        while True:
            try:
                capture = playback.get_next_capture()
                
                # Save depth frame
                if capture.depth is not None:
                    depth_frame_index += 1
                    depth_image = capture.depth  # Raw depth data
                    depth_group.create_dataset(f"frame_{depth_frame_index}", data=depth_image, compression="gzip")

                # Save IR frame
                if capture.ir is not None:
                    ir_frame_index += 1
                    ir_image = capture.ir  # Raw IR data
                    ir_group.create_dataset(f"frame_{ir_frame_index}", data=ir_image, compression="gzip")
                
            except EOFError:
                break

def main() -> None:
    filename = '../../../TestData/anonymous/ng3_t0_ks4_184.505_200.641_exo.mkv'
    output_hdf5_path = "../../../TestData/anonymous/ng3_t0_ks4/depth_ir_data.hdf5"

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)
    play(playback, output_hdf5_path)

    playback.close()

if __name__ == "__main__":
    main()
