#!/usr/bin/env python3

import os
import sys
import cv2
import csv
import pandas as pd

def process_video(video_path: str, ref_frame: int, ref_ts_ms: float, fps: float = 30.0):
    video_path = video_path.strip()
    if not os.path.isfile(video_path):
        print(f"[WARN] file not found: {video_path}")
        return

    base = os.path.dirname(video_path)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = os.path.join(base, f"{name}_timestamps.csv")
    print(f"→ Writing {out_csv}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_per_frame_ms = 1_000.0 / fps
    ref_zero = ref_frame - 1

    print(f"Total frames: {total_frames}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "epoch_ts_ms"])
        for frame_num in range(total_frames):
            ts = ref_ts_ms + (frame_num - ref_zero) * time_per_frame_ms
            # print(f"Frame {frame_num}: {ts:.2f} ms, {int(ts)} ms")
            writer.writerow([frame_num, int(ts)])
    cap.release()

def main(mapping_csv: str):
    df = pd.read_csv(mapping_csv)
    if not {"epoch_frame", "epoch_ts"}.issubset(df.columns):
        print("ERROR: CSV needs epoch_frame & epoch_ts columns")
        sys.exit(1)

    for _, row in df.iterrows():
        ref_frame = int(row["epoch_frame"])
        ref_ts    = float(row["epoch_ts"])
        # try vid_1, vid_2, vid_3 — skip if path missing or blank
        for i in (1, 2, 3):
            path_col = f"vid_{i}_path"
            raw = row.get(path_col, "")
            if pd.isna(raw):
                continue
            path = str(raw).strip()
            if not path:
                continue
            print("--" * 40)
            print(f"Processing {path} with ref_frame={ref_frame}, ref_ts={ref_ts}")
            process_video(path, ref_frame, ref_ts)
            print("--" * 40)
            # break
        # break
    



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_frame_timestamps.py <mapping_csv>")
        sys.exit(1)
    main(sys.argv[1])
