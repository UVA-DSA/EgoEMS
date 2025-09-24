#!/usr/bin/env python3
import cv2
import os
import subprocess
from pathlib import Path


# ---------- CONFIGURE HERE ----------
INPUT_VIDEO = r"G:\Research\EgoEMS\Dataset\EgoEMS_AAAI2026_Supplementary\Multimedia_Appendix\ng2_t0_modalities_final_short_clip.mp4"
OUTPUT_PATH = r"G:\Research\EgoEMS\Dataset\EgoEMS_AAAI2026_Supplementary\Multimedia_Appendix\ng2_t0_modalities_final_short_clip_title_fixed.mp4"
TEMP_VIDEO = str(Path(OUTPUT_PATH).with_suffix(".noaudio.mp4"))


# New titles to overlay
NEW_TITLE2 = "Smartwatch IMU Data"
NEW_TITLE3 = "ToF (Ground Truth Compression Depth)"  # example; set to what you want

# Title drawing params
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_FONT_SCALE = 0.6
TITLE_COLOR = (0, 0, 0)  # black text
TITLE_THICKNESS = 1
BOX_COLOR = (255, 255, 255)  # white cover
BOX_PADDING = 10

# These boxes should cover the existing titles for graph 2 and 3.
# Format: (x, y, w, h) in pixels. YOU MUST TUNE these to fit your video.
TITLE2_BOX = (850, 10, 1100, 35)  # example position for second graph title
TITLE3_BOX = (1475, 10, 800, 35)  # example position for third graph title

# Where to place the new titles (you can tweak the offsets relative to the boxes)
def title_position_from_box(box):
    x, y, w, h = box
    # Place text slightly inset
    return (x + 8, y + h - 10)

# -------------------------------

def mux_audio(video_no_audio_path, original_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio_path,
        "-i", original_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    print("Muxing audio with ffmpeg...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("ffmpeg failed:", res.stderr)
        raise RuntimeError("ffmpeg muxing failed")
    print("Final output with audio at:", output_path)

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Cannot open input video:", INPUT_VIDEO)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Failed to open writer:", TEMP_VIDEO)
        return

    print(f"Processing {total} frames...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Cover existing titles with white boxes
        x2, y2, w2, h2 = TITLE2_BOX
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), BOX_COLOR, -1)
        x3, y3, w3, h3 = TITLE3_BOX
        cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), BOX_COLOR, -1)

        # Draw new titles
        pos2 = title_position_from_box(TITLE2_BOX)
        cv2.putText(frame, NEW_TITLE2, pos2, FONT, TITLE_FONT_SCALE, TITLE_COLOR, TITLE_THICKNESS, cv2.LINE_AA)
        pos3 = title_position_from_box(TITLE3_BOX)
        cv2.putText(frame, NEW_TITLE3, pos3, FONT, TITLE_FONT_SCALE, TITLE_COLOR, TITLE_THICKNESS, cv2.LINE_AA)

        writer.write(frame)
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{total}")
        frame_idx += 1

    writer.release()
    cap.release()

    # Mux original audio back
    try:
        mux_audio(TEMP_VIDEO, INPUT_VIDEO, OUTPUT_PATH)
        # cleanup
        if os.path.exists(OUTPUT_PATH):
            os.remove(TEMP_VIDEO)
    except Exception as e:
        print("Audio mux failed, keeping no-audio output at:", TEMP_VIDEO)
        print("Error:", e)

if __name__ == "__main__":
    main()
