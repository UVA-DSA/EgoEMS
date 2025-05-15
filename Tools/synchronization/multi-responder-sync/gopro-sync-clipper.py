#!/usr/bin/env python3
import os
import csv
import subprocess
import sys

# === CONFIGURATION ===
SOURCE_FPS = 29.97002997002997    # your original FPS (only used for offset calc)
TARGET_FPS = 30.0                 # re‑encode to this framerate


def ffmpeg_clip(input_path: str, offset_sec: float, output_path: str):
    """
    Clip the input video starting at offset_sec, re‑encoding video to TARGET_FPS
    while copying audio.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset_sec:.3f}",
        "-i", input_path,
        # re‑encode video at 30fps:
        "-filter:v", f"fps={TARGET_FPS}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        # copy audio stream without re‑encoding:
        "-c:a", "copy",
        output_path
    ]
    print("🔹", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    
    # get input video paths from CSV from cmd line
    CSV_IN = sys.argv[1] if len(sys.argv) > 1 else CSV_IN
    
    with open(CSV_IN, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial = row["trial"]
            # gather vid_i paths & frame numbers
            vids, frames = [], []
            for i in (1, 2, 3):
                try:
                    pkey = f"vid_{i}_path"
                    fkey = f"vid_{i}_frame_num"
                    path = row.get(pkey, "").strip()
                    num  = row.get(fkey, "").strip()
                    if path and num:
                        vids.append(path)
                        frames.append(int(float(num)))
                except Exception as e:
                    print(f"⚠️  Trial {trial}: invalid video responder {i}, skipping.")
                    continue

            if not vids:
                print(f"⚠️  Trial {trial}: no videos found, skipping.")
                continue

            # compute offsets (s) relative to earliest frame
            min_frame   = min(frames)
            offsets_sec = [(f - min_frame) / SOURCE_FPS for f in frames]

            # clip & reencode each
            for idx, (in_path, offs) in enumerate(zip(vids, offsets_sec), start=1):
                video_base_path = os.path.dirname(in_path)
                base, _ = os.path.splitext(os.path.basename(in_path))
                video_name = os.path.basename(in_path).split(".")[0]
                print(f"🔹 Trial {trial} — video {idx} ({video_name})")
                print(f"Offset: {offs:.3f} s")
                
                out_name = f"{video_name}_synced.mp4"
                out_path = os.path.join(video_base_path, out_name)

                print(f"out path: {out_path}")
                if os.path.exists(out_path):
                    print(f"   ▶️  {out_name} exists — skipping.")
                else:
                    ffmpeg_clip(in_path, offs, out_path)
                    print(f"   ▶️ clipping {out_name} .")

            print(f"✅  Finished trial {trial}\n")
            break


if __name__ == "__main__":
    main()
