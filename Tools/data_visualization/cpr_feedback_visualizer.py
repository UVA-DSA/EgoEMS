#!/usr/bin/env python3
import cv2
import pandas as pd
import numpy as np
import subprocess
import os
import sys
from pathlib import Path

# ---------- CONFIGURE HERE ----------
VIDEO_PATH = r"G:\Research\EgoEMS\Dataset\EgoEMS_AAAI2026_Supplementary\Multimedia_Appendix\ng2_t0_modalities_final_short_clip_title_fixed.mp4"
CSV_PATH = "./ng2_visualization.csv"
OUTPUT_PATH = r"G:\Research\EgoEMS\Dataset\EgoEMS_AAAI2026_Supplementary\Multimedia_Appendix\feedback_annotated_with_audio.mp4"
# ------------------------------------

# ---------- Styling ----------
RATE_COLOR = (50, 200, 255)      # cyan-ish
DEPTH_COLOR = (255, 180, 50)     # orange-ish
BG_ALPHA = 0.6
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_RATE = 200.0
MAX_DEPTH = 50.0
# ----------------------------

def load_segments(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("start_t").reset_index(drop=True)
    # normalize "NA" feedback to empty
    df["rate_feedback"] = df["rate_feedback"].replace("NA", "")
    df["depth_feedback"] = df["depth_feedback"].replace("NA", "")
    return df.to_dict(orient="records")

def find_segment(segs, t, current_idx):
    n = len(segs)
    for i in range(current_idx, n):
        if segs[i]["start_t"] <= t < segs[i]["end_t"]:
            return i, segs[i]
        if t < segs[i]["start_t"]:
            break
    return None, None

def draw_panel(frame, topleft, size, title, entries, feedback, gt_feedback, color):
    x, y = topleft
    w, h = size
    overlay = frame.copy()
    # background
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 10, 10), -1)
    # border
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)

    pad = 8
    line_h = 8
    cur_y = y + pad + 8
    # title
    cv2.putText(frame, title, (x + pad, cur_y), FONT, 0.55, color, 1, cv2.LINE_AA)
    cur_y += int(line_h * 1.2)

    bar_w = w - 2 * pad - 120
    bar_h = 12
    for name, val, maxv in entries:
        cv2.putText(frame, f"{name}", (x + pad, cur_y + bar_h - 2), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
        bar_x = x + pad + 90
        bar_y = cur_y
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        norm = float(val) / maxv if maxv > 0 else 0
        norm = max(0.0, min(1.0, norm))
        filled = int(bar_w * norm)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
        txt = f"{val:.1f}" if isinstance(val, (float,)) else str(val)
        cv2.putText(frame, txt, (bar_x + bar_w + 5, bar_y + bar_h - 2), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
        cur_y += bar_h + 8

    fb_y = y + h - pad - 30
    cv2.putText(frame, "Feedback:", (x + pad, fb_y), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    fb_text = feedback if feedback and pd.notna(feedback) else ""
    fb_box_w = w - 2 * pad
    fb_box_h = 24
    fb_box_x = x + pad
    fb_box_y = fb_y + 5

    # check if gt_feedback matches feedback. if it matches, font color is green, else red
    if gt_feedback and pd.notna(gt_feedback):
        if gt_feedback == feedback:
            UPDATED_TEXT_COLOR = (0, 255, 0)  # green
        else:
            UPDATED_TEXT_COLOR = (0, 0, 255)  # red
    else:
        UPDATED_TEXT_COLOR = (255, 255, 255)  # white
    cv2.rectangle(frame, (fb_box_x, fb_box_y), (fb_box_x + fb_box_w, fb_box_y + fb_box_h), (50, 50, 50), -1)
    cv2.putText(frame, fb_text, (fb_box_x + 6, fb_box_y + fb_box_h - 6), FONT, 0.5, UPDATED_TEXT_COLOR, 1, cv2.LINE_AA)

def draw_timeline(frame, t, duration):
    h, w = frame.shape[:2]
    bar_h = 8
    margin = 20
    x1 = margin
    x2 = w - margin
    y = h - 30
    cv2.rectangle(frame, (x1, y), (x2, y + bar_h), (50, 50, 50), -1)
    if duration > 0:
        progress = min(1.0, max(0.0, t / duration))
        filled_w = int((x2 - x1) * progress)
        cv2.rectangle(frame, (x1, y), (x1 + filled_w, y + bar_h), (200, 200, 200), -1)
    time_str = f"{t:.1f}s / {duration:.1f}s"
    cv2.putText(frame, time_str, (x1, y - 10), FONT, 0.6, (220, 220, 220), 2, cv2.LINE_AA)


def draw_gt_cpr_box(frame, seg):
    H, W = frame.shape[:2]
    box_w = 180
    box_h = 50
    x = W - box_w - 15
    y = 80
    overlay = frame.copy()
    # background panel (semi-opaque)
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    # border
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (200, 200, 200), 2)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title labels and values
    pad = 10
    rate_text = f"GT CPR Rate: {seg.get('gt_rate', 0):.1f}"
    depth_text = f"GT CPR Depth: {seg.get('gt_depth', 0):.1f}"
    # Larger text
    cv2.putText(frame, rate_text, (x + pad, y + 20), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, depth_text, (x + pad, y + 40), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def annotate_frame(frame, seg, t, total_dur, frame_idx):
    H, W = frame.shape[:2]
    rate_entries = [
        ("GT Rate", seg["gt_rate"], MAX_RATE),
        ("Smartwatch", seg["smartwatch_rate"], MAX_RATE),
        ("Ego", seg["ego_rate"], MAX_RATE),
        ("Fusion", seg["fusion_rate"], MAX_RATE),
    ]
    draw_panel(frame, (15, 15), (210, 155), "Rate", rate_entries, seg.get("rate_feedback", ""), seg.get("gt_rate_feedback", ""),RATE_COLOR)

    depth_entries = [
        ("GT Depth", seg["gt_depth"], MAX_DEPTH),
        ("Smartwatch", seg["smartwatch_depth"], MAX_DEPTH),
        ("Ego", seg["ego_depth"], MAX_DEPTH),
        ("Fusion", seg["fusion_depth"], MAX_DEPTH),
    ]
    draw_panel(frame, (410, 15),  (210, 155), "Depth", depth_entries, seg.get("depth_feedback", ""), seg.get("gt_depth_feedback", ""), DEPTH_COLOR)

    draw_gt_cpr_box(frame, seg)
    # draw_timeline(frame, t, total_dur)
    # cv2.putText(frame, f"Frame {frame_idx}", (W - 180, H - 10), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

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
        print("ffmpeg mux failed:", res.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg muxing failed")
    print("Final video written to:", output_path)

def main():
    segs = load_segments(CSV_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH, file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = str(Path(OUTPUT_PATH).with_suffix(".noaudio.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Failed to open writer:", temp_video, file=sys.stderr)
        sys.exit(1)

    current_seg_idx = 0
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        idx, seg = find_segment(segs, t, current_seg_idx)
        if seg is not None:
            current_seg_idx = idx
            annotate_frame(frame, seg, t, duration, frame_idx)
        else:
            draw_timeline(frame, t, duration)
            H, W = frame.shape[:2]
            cv2.putText(frame, f"Frame {frame_idx}", (W - 180, H - 10), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(frame)
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames} (t={t:.2f}s)")

    writer.release()
    cap.release()

    try:
        mux_audio(temp_video, VIDEO_PATH, OUTPUT_PATH)
    except Exception:
        print("Muxing failed; output without audio is at", temp_video, file=sys.stderr)
        sys.exit(1)
    else:
        if os.path.exists(OUTPUT_PATH):
            try:
                os.remove(temp_video)
            except Exception:
                pass

if __name__ == "__main__":
    main()
