import pandas as pd
import math
# get command line arguments
import argparse


if __name__ == "__main__":
    # get cmd line args
    parser = argparse.ArgumentParser(description="Downsample a GoPro timestamped CSV file to 30 FPS")
    parser.add_argument("file_path", type=str, help="Path to the input CSV file")
    args = parser.parse_args()
    file_path = args.file_path


    # Load the CSV file
    df = pd.read_csv(file_path)

    # Calculate the frame interval for 30 FPS
    interval = math.ceil(59.97 / 30)
    print(f"Frame interval for 30 FPS: {int(interval)}")

    # Select every `interval` frame
    df_30fps = df.iloc[::int(interval)].reset_index(drop=True)

    print(df_30fps.head())
    # outputpath, append to original file name
    output_path = file_path.replace(".csv", "_30fps.csv")
    df_30fps.to_csv(output_path, index=False)

    print(f"Downsampled data saved to {output_path}")
