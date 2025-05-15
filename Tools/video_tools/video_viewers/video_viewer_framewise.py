import cv2
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

# === CONFIG ===
VIDEO_CSV = "/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Tools/file_handling/opvrs_data_mappings.csv"
VIDEO_COLUMN = "GoPro_Path"
WINDOW_SIZE = (1280, 720)
FRAME_STEP = 5       # frames to skip on fine control
PLAY_DELAY = 30      # ms between frames when playing

class VideoPlayer:
    def __init__(self, master, video_paths):
        self.master = master
        self.video_paths = video_paths
        self.idx = 0
        self.cap = None
        self.playing = False
        self.photo = None

        # --- UI ---
        self.video_label = tk.Label(master)
        self.video_label.pack()

        ctrl = tk.Frame(master)
        ctrl.pack(fill="x", pady=5)

        tk.Button(ctrl, text="⏮ Prev Video", command=self.prev_video).pack(side="left")
        tk.Button(ctrl, text="⏪ Back 5",     command=self.back_5).pack(side="left")
        self.play_btn = tk.Button(ctrl, text="▶ Play",  command=self.toggle_play)
        self.play_btn.pack(side="left", padx=10)
        tk.Button(ctrl, text="Forward 5 ⏩", command=self.forward_5).pack(side="left")
        tk.Button(ctrl, text="Next Video ⏭", command=self.next_video).pack(side="left")
        tk.Button(ctrl, text="✖ Quit",       command=self.quit).pack(side="right")

        self.open_video(self.video_paths[self.idx])

    def open_video(self, path):
        # release old if exists
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        self.master.title(f"Video {self.idx+1}/{len(self.video_paths)}: {path.split('/')[-1]}")
        # show first frame paused
        self.playing = False
        self.play_btn.config(text="▶ Play")
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # loop or just pause
            self.playing = False
            self.play_btn.config(text="▶ Play")
            return

        # draw frame number
        fn = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(frame, f"Frame: {fn}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # resize & convert for Tkinter
        frame = cv2.resize(frame, WINDOW_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.master.after(PLAY_DELAY, self.play_loop)

    def play_loop(self):
        if not self.playing:
            return
        self.show_frame()
        self.master.after(PLAY_DELAY, self.play_loop)

    def back_5(self):
        self.playing = False
        self.play_btn.config(text="▶ Play")
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        new = max(0, cur - FRAME_STEP)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new)
        self.show_frame()

    def forward_5(self):
        self.playing = False
        self.play_btn.config(text="▶ Play")
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        new = cur + FRAME_STEP
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new)
        self.show_frame()

    def next_video(self):
        self.idx = (self.idx + 1) % len(self.video_paths)
        self.open_video(self.video_paths[self.idx])

    def prev_video(self):
        self.idx = (self.idx - 1) % len(self.video_paths)
        self.open_video(self.video_paths[self.idx])

    def quit(self):
        self.cap.release()
        self.master.destroy()


def load_video_list(csv_path, column):
    df = pd.read_csv(csv_path)
    paths = []
    for p in df[column].dropna():
        p = str(p).strip()
        # adapt if you need to force "_synced.mp4":
        p = p.replace(".MP4", "_synced.mp4")
        paths.append(p)
    return paths

if __name__ == "__main__":
    video_paths = load_video_list(VIDEO_CSV, VIDEO_COLUMN)
    if not video_paths:
        raise RuntimeError("No video paths found in CSV!")

    root = tk.Tk()
    player = VideoPlayer(root, video_paths)
    root.mainloop()
