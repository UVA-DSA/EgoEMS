#!/usr/bin/env python3
import json
import argparse
import os
import cv2

def restructure(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    output = {}

    for subj in data.get("subjects", []):
        sid = subj.get("subject_id")
        for scenario in subj.get("scenarios", []):
            scid = scenario.get("scenario_id")
            for trial in scenario.get("trials", []):
                tid = trial.get("trial_id")

                ego_video_path = trial.get('streams')['egocam_rgb_audio'].get('file_path')
                print(f"Processing {ego_video_path}...")

                # use opencv to get number of frames and fps
                cap = cv2.VideoCapture(ego_video_path)
                duration_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration_second = duration_frame / fps

                print(f"  → {duration_frame} frames, {fps} fps, {duration_second:.2f} seconds")

                key = ego_video_path

                # key = f"{sid}_{scid}_t{tid}_ego_final.mp4"

                annots = []
                for ks in trial.get("keysteps", []):
                    st = ks.get("start_t")
                    et = ks.get("end_t")
                    lbl = ks.get("label")
                    if st is not None and et is not None and lbl is not None:
                        annots.append({
                            "segment": [st, et],
                            "label": lbl
                        })

                output[key] = {
                    "duration_second": duration_second,
                    "duration_frame":  duration_frame,
                    "annotations":       annots,
                    "feature_frame":     duration_frame,
                    "fps":               fps,
                    "rfps":              fps
                }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Restructured {len(output)} videos → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Restructure your trial JSON into per-video entries.")
    p.add_argument("-i", "--input",  required=True,
                   help="Path to your original JSON")
    p.add_argument("-o", "--output", required=True,
                   help="Where to write the restructured JSON")
    args = p.parse_args()
    restructure(args.input, args.output)