import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import requests


# Input source configuration.
USE_WEBCAM = False
WEBCAM_INDEX = 1

# Set this to the video you want to test when USE_WEBCAM is False.
VIDEO_PATH = Path("/mnt/f/repos/EgoEMS/Tools/inference/videos/ms1_cardiac_arrest_t4_ks2_5.273_12.523_ego.mp4")

# Server settings.
SERVER_URL = "http://localhost:8000/infer"
REQUEST_TIMEOUT_SECONDS = 30

# Optional throttles for easier debugging.
MAX_FRAMES = None
SEND_EVERY_NTH_FRAME = 1
JPEG_QUALITY = 90
SHOW_PREVIEW = False
PREVIEW_WINDOW_NAME = "Server Test Client"


def format_detections(detections: List[Dict[str, Any]]) -> str:
    if not detections:
        return "no detections"

    parts = []
    for det in detections:
        label = det.get("label", "unknown")
        score = det.get("score", 0.0)
        box = det.get("box_xyxy", [])
        parts.append(f"{label}({score:.2f}) {box}")
    return "; ".join(parts)


def send_frame(session: requests.Session, frame_bgr, frame_id: int) -> Dict[str, Any]:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Failed to encode frame {frame_id} as JPEG.")

    headers = {
        "Content-Type": "application/octet-stream",
        "x-frame-id": str(frame_id),
        "x-timestamp": f"{time.time():.6f}",
    }

    response = session.post(
        SERVER_URL,
        data=encoded.tobytes(),
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def open_capture() -> cv2.VideoCapture:
    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {WEBCAM_INDEX}.")
        return cap

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    return cap


def main() -> None:
    if SEND_EVERY_NTH_FRAME <= 0:
        raise ValueError("SEND_EVERY_NTH_FRAME must be > 0")
    if MAX_FRAMES is not None and MAX_FRAMES <= 0:
        raise ValueError("MAX_FRAMES must be > 0 when provided")

    cap = open_capture()

    if USE_WEBCAM:
        print(f"[client] webcam index: {WEBCAM_INDEX}")
        print("[client] press 'q' in the preview window to stop")
    else:
        print(f"[client] video: {VIDEO_PATH}")
    print(f"[client] server: {SERVER_URL}")

    frame_idx = 0
    sent_count = 0
    start_time = time.perf_counter()

    session = requests.Session()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % SEND_EVERY_NTH_FRAME != 0:
                continue

            result = send_frame(session=session, frame_bgr=frame, frame_id=frame_idx)
            sent_count += 1

            detections = result.get("detections", [])
            inference_ms = result.get("inference_ms", -1.0)
            preprocess_ms = result.get("preprocess_ms", -1.0)
            postprocess_ms = result.get("postprocess_ms", -1.0)
            print(
                f"[frame {frame_idx}] "
                f"pre={preprocess_ms:.2f} ms "
                f"infer={inference_ms:.2f} ms "
                f"post={postprocess_ms:.2f} ms | "
                f"{format_detections(detections)}"
            )

            if SHOW_PREVIEW:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"frame={frame_idx} sent={sent_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if MAX_FRAMES is not None and sent_count >= MAX_FRAMES:
                break
    finally:
        cap.release()
        session.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    fps = sent_count / elapsed if elapsed > 0 else 0.0
    print(f"[client] sent {sent_count} frames in {elapsed:.2f}s ({fps:.2f} req/s)")


if __name__ == "__main__":
    main()
