import os
# Must come before any cv2 import
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import av
import cv2
import pandas as pd

skip_frames = 5
CSV_OUT = "frame_positions.csv"
CSV_IN  = "./grouped_gopro_paths.csv"

class VideoPlayer:
    def __init__(self, path):
        self.path      = path
        self.container = av.open(path)
        self.stream    = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self.fps          = float(self.stream.average_rate) or 25.0
        self.total_frames = self.stream.frames or 0

        self.playing      = False
        self.current_frame= None
        self.frame_no     = 0

        self._reset_iter()
        self.step()

    def _reset_iter(self):
        self.container.seek(0)
        self.dec_iter = self.container.decode(video=0)
        self.frame_no = 0

    def step(self):
        try:
            frame = next(self.dec_iter)
        except (StopIteration, av.AVError):
            # real EOF → loop
            self._reset_iter()
            frame = next(self.dec_iter)
        else:
            self.frame_no += 1
        # keep BGR for cv2→PIL conversion
        self.current_frame = frame.to_ndarray(format="bgr24")

    def toggle_play(self):
        self.playing = not self.playing

    def set_play(self, s: bool):
        self.playing = s

    def stop(self):
        self.playing = False
        self._reset_iter()
        self.step()

    def seek(self, delta: int):
        target = max(self.frame_no + delta, 0)
        ts = int(target / self.fps * av.time_base)
        # seek by timestamp
        self.container.seek(ts, any_frame=False, backward=True, stream=self.stream)
        self._reset_iter()
        # fast‑forward to target
        for _ in range(target):
            try: next(self.dec_iter)
            except: break
        self.frame_no = target
        frame = next(self.dec_iter)
        self.current_frame = frame.to_ndarray(format="bgr24")

    def close(self):
        self.container.close()


class VideoApp:
    def __init__(self, root):
        self.root    = root
        self.scale   = 0.3
        
        self.master_play = False


        # load all trials
        df_in = pd.read_csv(CSV_IN)
        trials = []
        for _, row in df_in.iterrows():
            vid_list = [row[c] for c in df_in.columns
                        if c.startswith("GoPro_") and pd.notna(row[c])]
            trials.append({
                "trial": int(row["Main_Trial"]),
                "videos": vid_list
            })

        # read processed trials
        if os.path.exists(CSV_OUT):
            df_done = pd.read_csv(CSV_OUT)
            done = set(df_done["trial"].astype(int).tolist())
        else:
            done = set()

        # keep only unprocessed
        self.remaining = [t for t in trials if t["trial"] not in done]
        if not self.remaining:
            messagebox.showinfo("Nothing to do", "All trials already processed.")
            root.destroy()
            return

        # UI containers
        root.title("Synchronized Video Sync")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.video_frame   = ttk.Frame(root)
        self.video_frame.pack(padx=10, pady=10)
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(pady=10)

        # control buttons
        self.btn_master = ttk.Button(self.control_frame,
                                     text="Master Play/Pause",
                                     command=self._master_toggle)
        self.btn_master.pack(side="left", padx=5)

        self.btn_save   = ttk.Button(self.control_frame,
                                     text="Save Frames",
                                     command=self._save)
        self.btn_save.pack(side="left", padx=5)

        self.btn_rewind = ttk.Button(self.control_frame,
                                     text="Master Rewind",
                                     command=self._master_rewind)
        self.btn_rewind.pack(side="left", padx=5)

        # playback parameters (all same FPS)
        sample_vid = self.remaining[0]["videos"][0]
        dummy = av.open(sample_vid).streams.video[0]
        self.fps = float(dummy.average_rate) or 25.0
        dummy.container.close()

        # for update loop
        self.current_idx = 0
        self.players     = []
        self.panels      = []
        self.labels      = []

        # load first trial
        self._load_trial()

        # start the sync loop
        delay = int(1000 / self.fps)
        self._loop_id = root.after(delay, self._update)

    def _load_trial(self):
        # clear old players & UI
        for p in self.players:
            p.close()
        for w in self.video_frame.winfo_children():
            w.destroy()
        self.players.clear()
        self.panels.clear()
        self.labels.clear()

        # set window title to show trial
        t = self.remaining[self.current_idx]["trial"]
        self.root.title(f"Synchronized Video Sync — Trial {t}")

        # build new players
        vids = self.remaining[self.current_idx]["videos"]
        for col, path in enumerate(vids):
            vp = VideoPlayer(path)
            self.players.append(vp)

            frm = ttk.Frame(self.video_frame, relief="sunken", borderwidth=1)
            frm.grid(row=0, column=col, padx=5, pady=5)

            lbl = tk.Label(frm)
            lbl.pack()
            self.panels.append(lbl)

            fl = tk.Label(frm, text="Frame: 0")
            fl.pack(pady=2)
            self.labels.append(fl)

            ctr = ttk.Frame(frm)
            ctr.pack(pady=5)
            ttk.Button(ctr, text="Play/Pause", command=vp.toggle_play).pack(fill="x")
            ttk.Button(ctr, text="Stop",       command=vp.stop).pack(fill="x", pady=2)
            ttk.Button(ctr, text="Forward",    command=lambda v=vp: v.seek(skip_frames)).pack(fill="x")
            ttk.Button(ctr, text="Backward",   command=lambda v=vp: v.seek(-skip_frames)).pack(fill="x", pady=2)

    def _master_toggle(self):
        # toggle all
        state = not getattr(self, "master_play", False)
        self.master_play = state
        for p in self.players:
            p.set_play(state)

    def _master_rewind(self):
        # stop & rewind all
        self.master_play = False
        for p in self.players:
            p.stop()

    def _save(self):
        trial_no = self.remaining[self.current_idx]["trial"]
        # build row
        cols, vals = ["trial"], [trial_no]
        for i, p in enumerate(self.players, start=1):
            cols += [f"vid_{i}_path", f"vid_{i}_frame_num"]
            vals += [p.path, p.frame_no]

        df = pd.DataFrame([vals], columns=cols)
        df.to_csv(CSV_OUT, mode="a", header=not os.path.exists(CSV_OUT), index=False)

        messagebox.showinfo("Saved", f"Successfully saved frames for trial {trial_no}")

        # advance to next
        self.current_idx += 1
        if self.current_idx >= len(self.remaining):
            messagebox.showinfo("All done", "You’ve processed all remaining trials.")
            self.btn_save.config(state="disabled")
            self.btn_master.config(state="disabled")
            self.btn_rewind.config(state="disabled")
        else:
            self._load_trial()

    def _update(self):
        for lbl, fl, p in zip(self.panels, self.labels, self.players):
            
            # now honor BOTH master and individual play/pause
            if self.master_play or p.playing:
                p.step()

            frm = p.current_frame
            if frm is not None:
                img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
                tkimg = ImageTk.PhotoImage(image=Image.fromarray(img))
                lbl.imgtk = tkimg
                lbl.config(image=tkimg)
                fl.config(text=f"Frame: {p.frame_no}")

        delay = int(1000 / self.fps)
        self._loop_id = self.root.after(delay, self._update)

    def _on_close(self):
        self.root.after_cancel(self._loop_id)
        for p in self.players:
            p.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = VideoApp(root)
    root.mainloop()
