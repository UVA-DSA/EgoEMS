#!/usr/bin/env python3
import sys, cv2, av, os
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QSlider, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QRubberBand, QWidget,
    QProgressDialog, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QRect, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap

class ROISelector(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw ROI")
        rgb = frame[..., ::-1]
        h, w, _ = rgb.shape
        bpl = 3 * w
        qimg = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        lbl = QLabel(self)
        lbl.setPixmap(pix)
        lbl.setFixedSize(w, h)
        self.band = QRubberBand(QRubberBand.Rectangle, lbl)
        self.origin = QPoint()
        self.rect = None
        lbl.mousePressEvent    = self.onPress
        lbl.mouseMoveEvent     = self.onMove
        lbl.mouseReleaseEvent  = self.onRelease
        layout = QVBoxLayout(self)
        layout.addWidget(lbl)
    def onPress(self, ev):
        if ev.button() == Qt.LeftButton:
            self.origin = ev.pos()
            self.band.setGeometry(QRect(self.origin, QSize()))
            self.band.show()
    def onMove(self, ev):
        if not self.origin.isNull():
            self.band.setGeometry(QRect(self.origin, ev.pos()).normalized())
    def onRelease(self, ev):
        if ev.button() == Qt.LeftButton:
            self.band.hide()
            r = self.band.geometry()
            self.rect = (r.x(), r.y(), r.width(), r.height())
            self.accept()

class FaceBlurApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EgoEMS - Video Deidentification Verification & Correction Tool")
        self.resize(800, 600)
        self.cap = None; self.total = 0; self.fps = 30; self.current = 0
        self.video_path = ""; self.video_name = ""
        self.rois = []; self.start_frame = None; self.blurred_frames = {}; self.saved = False

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False); self.slider.sliderMoved.connect(self.seek)

        load_btn   = QPushButton("Load Video");      load_btn.clicked.connect(self.load)
        self.play_btn   = QPushButton("Play");       self.play_btn.clicked.connect(self.play)
        self.prev3_btn  = QPushButton("◀◀");         self.prev3_btn.clicked.connect(lambda: self.show_frame(self.current-3))
        self.prev_btn   = QPushButton("⯇");          self.prev_btn.clicked.connect(lambda: self.show_frame(self.current-1))
        self.next_btn   = QPushButton("▶");          self.next_btn.clicked.connect(lambda: self.show_frame(self.current+1))
        self.next3_btn  = QPushButton("▶▶");         self.next3_btn.clicked.connect(lambda: self.show_frame(self.current+3))
        self.mark_btn   = QPushButton("Mark ROI");   self.mark_btn.clicked.connect(self.mark_roi)
        self.track_btn  = QPushButton("Track ROI & Blur"); self.track_btn.clicked.connect(self.track_blur)
        self.track_back_btn = QPushButton("Track ROI Backwards & Blur"); self.track_back_btn.clicked.connect(self.track_blur_backward)
        self.save_btn   = QPushButton("Save Video"); self.save_btn.clicked.connect(self.save)
        self.close_btn  = QPushButton("Close");      self.close_btn.clicked.connect(self.close_app)

        # Dropdown for tracking duration
        self.duration_combo = QComboBox()
        self.duration_combo.addItem("2 seconds", 2)
        self.duration_combo.addItem("5 seconds", 5)
        self.duration_combo.addItem("10 seconds", 10)
        self.duration_combo.setCurrentIndex(0)
        duration_label = QLabel("Track Duration:")

        ctrl = QHBoxLayout()
        for w in (load_btn, self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                  self.mark_btn, duration_label, self.duration_combo, self.track_btn, self.track_back_btn, self.save_btn, self.close_btn):
            ctrl.addWidget(w)
        root = QWidget(); lay = QVBoxLayout(root); lay.addWidget(self.video_label, 1); lay.addWidget(self.slider); lay.addLayout(ctrl)
        self.setCentralWidget(root)
        self.timer = QTimer(self); self.timer.timeout.connect(self.next_frame)
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.track_back_btn, self.save_btn, self.close_btn):
            btn.setEnabled(False)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path: return
        if self.cap: self.cap.release()
        self.video_path = path
        self.video_name = os.path.splitext(os.path.basename(path))[0]
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video"); return
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.slider.setRange(0, self.total-1); self.slider.setEnabled(True)
        self.current = 0; self.rois = []; self.start_frame = None; self.blurred_frames.clear(); self.saved = False
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.track_back_btn, self.save_btn, self.close_btn):
            btn.setEnabled(True)
        self.show_frame(0)

    def show_frame(self, idx):
        if not self.cap or not (0 <= idx < self.total): return
        self.current = idx; self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret: return
        frame = self.blurred_frames.get(idx, frame)
        self._display(frame)
        self.slider.blockSignals(True); self.slider.setValue(idx); self.slider.blockSignals(False)

    def _display(self, frame):
        disp = frame.copy()
        h, w = disp.shape[:2]
        text = f"Frame {self.current}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        x = w - text_w - 10
        y = 10 + text_h
        cv2.putText(disp, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(disp, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        rgb = disp[..., ::-1]; bpl = 3 * w
        qimg = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def seek(self, pos):
        if self.timer.isActive(): self.play()
        self.show_frame(pos)

    def play(self):
        if not self.cap: return
        if self.timer.isActive():
            self.timer.stop(); self.play_btn.setText("Play")
        else:
            if self.current >= self.total - 1: self.show_frame(0)
            interval = int(1000/self.fps); self.timer.start(interval); self.play_btn.setText("Pause")

    def next_frame(self):
        if self.current < self.total-1: self.show_frame(self.current+1)
        else: self.play()

    def mark_roi(self):
        if self.timer.isActive(): self.play()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current); ret, frame = self.cap.read()
        if not ret: return
        dlg = ROISelector(frame, self)
        if dlg.exec_() == QDialog.Accepted and dlg.rect:
            bbox = tuple(dlg.rect); self.start_frame = self.current
            self.rois.append((self.start_frame, bbox))
            QMessageBox.information(self, "ROI Marked", f"ROI marked at frame {self.start_frame}.")
            x, y, w, h = bbox; cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self._display(frame)

    def _get_track_frames(self):
        secs = self.duration_combo.currentData()
        if secs is None:
            secs = 2
        return int(round(secs * self.fps))

    @staticmethod
    def apply_blur(frame, box):
        x, y, w, h = map(int, box)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            # pixelate: downscale then upscale with nearest neighbor
            h_roi, w_roi = roi.shape[:2]
            # choose block size so that the smaller dimension is reduced to ~16 pixels (adjustable)
            scale = max(1, min(h_roi, w_roi) // 4)
            small_w = max(1, w_roi // scale)
            small_h = max(1, h_roi // scale)
            temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = pixelated
            return True
        return False

    def track_blur(self):
        if not self.rois:
            QMessageBox.warning(self, "No ROI", "Please mark at least one ROI first."); return
        selected_secs = self.duration_combo.currentData()
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.track_back_btn):
            btn.setEnabled(False)
        QMessageBox.information(self, "Tracking", f"Running forward tracker for {selected_secs} seconds…"); QApplication.processEvents()
        to_track = self._get_track_frames()
        start_frames = [sf for sf, _ in self.rois]
        min_sf = min(start_frames)
        max_ef = max(sf + to_track for sf, _ in self.rois)
        max_end_frame = min(max_ef, self.total - 1)
        cap2 = cv2.VideoCapture(self.video_path); cap2.set(cv2.CAP_PROP_POS_FRAMES, min_sf)
        active_trackers = []
        for frame_idx in range(min_sf, max_end_frame + 1):
            ret, raw_frame = cap2.read()
            if not ret: break
            # base frame is existing blurred if present
            frame = self.blurred_frames.get(frame_idx, raw_frame).copy()
            did_blur = False
            for start_frame, bbox in self.rois:
                if start_frame == frame_idx:
                    init_frame = self.blurred_frames.get(frame_idx, raw_frame).copy()
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(init_frame, bbox)
                    active_trackers.append({'tracker': tracker, 'end_frame': start_frame + to_track})
            for active in active_trackers:
                if frame_idx < active['end_frame']:
                    ok, box = active['tracker'].update(frame)
                    if ok:
                        applied = self.apply_blur(frame, box)
                        if applied:
                            did_blur = True
            if did_blur:
                self.blurred_frames[frame_idx] = frame.copy()
            if frame_idx % 10 == 0 and did_blur:
                self._display(frame); QApplication.processEvents()
        cap2.release()
        next_frame_to_show = min(max_end_frame + 1, self.total - 1)
        self.show_frame(next_frame_to_show)
        QMessageBox.information(self, "Completed", "Forward tracking and blurring done.")
        self.rois.clear(); self.saved = False
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.track_back_btn, self.save_btn):
            btn.setEnabled(True)

    def track_blur_backward(self):
        if not self.rois:
            QMessageBox.warning(self, "No ROI", "Please mark at least one ROI first."); return
        selected_secs = self.duration_combo.currentData()
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.track_back_btn):
            btn.setEnabled(False)
        QMessageBox.information(self, "Tracking", f"Running backward tracker for {selected_secs} seconds…"); QApplication.processEvents()
        to_track = self._get_track_frames()
        cap2 = cv2.VideoCapture(self.video_path)
        earliest_frame_after = self.total - 1
        for start_frame, bbox in self.rois:
            start_bkw = max(0, start_frame - to_track)
            frames = []
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start_bkw)
            for fi in range(start_bkw, start_frame + 1):
                ret, raw = cap2.read()
                if not ret: break
                base = self.blurred_frames.get(fi, raw).copy()
                frames.append((fi, base))
            if not frames:
                continue
            reversed_frames = list(reversed(frames))
            _, first_frame = reversed_frames[0]
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame.copy(), bbox)
            for rev_idx, (orig_frame_idx, frame) in enumerate(reversed_frames):
                working = frame.copy()
                did_blur = False
                if rev_idx == 0:
                    applied = self.apply_blur(working, bbox)
                    if applied:
                        did_blur = True
                else:
                    ok, box = tracker.update(working)
                    if ok:
                        applied = self.apply_blur(working, box)
                        if applied:
                            did_blur = True
                if did_blur:
                    self.blurred_frames[orig_frame_idx] = working.copy()
                if orig_frame_idx < earliest_frame_after:
                    earliest_frame_after = orig_frame_idx
                if rev_idx % 10 == 0 and did_blur:
                    self._display(working); QApplication.processEvents()
        cap2.release()
        frame_to_show = earliest_frame_after if earliest_frame_after < self.total else 0
        self.show_frame(frame_to_show)
        QMessageBox.information(self, "Completed", "Backward tracking and blurring done.")
        self.rois.clear(); self.saved = False
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.track_back_btn, self.save_btn):
            btn.setEnabled(True)

    def save(self):
        if not self.blurred_frames:
            QMessageBox.warning(self, "Nothing to Save", "No blurred frames have been generated.")
            return

        # get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_dir = os.path.dirname(self.video_path)
        default_fname = f"Verified_{self.video_name}_{timestamp}.mp4"
        default_path = os.path.join(orig_dir, default_fname)
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", default_path, "MP4 Files (*.mp4)")
        if not path: return

        in_container = None
        out_container = None
        progress = None
        try:
            in_container = av.open(self.video_path)
            in_video_stream = in_container.streams.video[0]
            in_audio_streams = in_container.streams.audio
            total_frames = in_video_stream.frames or int(in_video_stream.duration * in_video_stream.average_rate)

            progress = QProgressDialog("Saving video...", "Cancel", 0, total_frames, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(False); progress.setAutoReset(False)
            progress.show()

            out_container = av.open(path, mode='w')
            out_video_stream = out_container.add_stream("libx264", rate=in_video_stream.average_rate)
            out_video_stream.width = in_video_stream.width
            out_video_stream.height = in_video_stream.height
            out_video_stream.pix_fmt = 'yuv420p'
            out_video_stream.options = {'crf': '20'}

            out_audio_streams = []
            if in_audio_streams:
                out_audio_streams = [out_container.add_stream(template=s) for s in in_audio_streams]

            frame_idx = 0
            was_cancelled = False

            for packet in in_container.demux(in_video_stream, *in_audio_streams):
                if progress.wasCanceled():
                    was_cancelled = True; break

                if packet.stream in in_audio_streams:
                    idx = in_audio_streams.index(packet.stream)
                    packet.stream = out_audio_streams[idx]
                    out_container.mux(packet)
                    continue

                if packet.stream == in_video_stream:
                    for frame in packet.decode():
                        frame_to_encode = frame
                        numpy_blurred_frame = self.blurred_frames.get(frame_idx)
                        if numpy_blurred_frame is not None:
                            new_frame = av.VideoFrame.from_ndarray(numpy_blurred_frame, format='bgr24')
                            new_frame.pts = frame.pts
                            new_frame.time_base = frame.time_base
                            frame_to_encode = new_frame

                        for out_packet in out_video_stream.encode(frame_to_encode):
                            out_container.mux(out_packet)

                        frame_idx += 1
                        progress.setValue(frame_idx)

            if not was_cancelled:
                for out_packet in out_video_stream.encode():
                    out_container.mux(out_packet)

            if progress:
                progress.close()

            if was_cancelled:
                QMessageBox.information(self, "Cancelled", "Save operation was cancelled.")
            else:
                self.saved = True
                QMessageBox.information(self, "Saved", f"Video with audio successfully written to\n{path}")

        except Exception as e:
            if progress:
                progress.close()
            # Primary save failed; attempt fallback to temp file (no audio) with timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_name = f"{self.video_name}_fallback_{ts}.mp4"
            fallback_path = os.path.join(orig_dir, fallback_name)
            try:
                # Open original capture to iterate frames
                cap_fb = cv2.VideoCapture(self.video_path)
                if not cap_fb.isOpened():
                    raise RuntimeError("Cannot open video for fallback write.")
                width = int(cap_fb.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_fb.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap_fb.get(cv2.CAP_PROP_FPS) or self.fps
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(fallback_path, fourcc, fps, (width, height))
                total = int(cap_fb.get(cv2.CAP_PROP_FRAME_COUNT))
                fb_progress = QProgressDialog("Fallback saving (no audio)...", "Cancel", 0, total, self)
                fb_progress.setWindowModality(Qt.WindowModal)
                fb_progress.setAutoClose(False); fb_progress.setAutoReset(False)
                fb_progress.show()
                frame_idx = 0
                while True:
                    ret, frame = cap_fb.read()
                    if not ret:
                        break
                    out_frame = self.blurred_frames.get(frame_idx, frame)
                    writer.write(out_frame)
                    frame_idx += 1
                    fb_progress.setValue(frame_idx)
                    QApplication.processEvents()
                    if fb_progress.wasCanceled():
                        break
                fb_progress.close()
                writer.release()
                cap_fb.release()
                self.saved = True
                QMessageBox.warning(self, "Partial Save",
                                    f"Primary save failed ({e}).\n"
                                    f"Fallback video without audio written to:\n{fallback_path}")
            except Exception as fb_e:
                QMessageBox.critical(self, "Save Failed",
                                     f"Primary save error: {e}\n"
                                     f"Fallback also failed: {fb_e}")
        finally:
            if in_container: in_container.close()
            if out_container: out_container.close()
            # If primary was cancelled and its file exists, remove it
            if 'was_cancelled' in locals() and was_cancelled and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def close_app(self):
        if self.blurred_frames and not self.saved:
            resp = QMessageBox.question(self, "Unsaved Changes", "Save before closing?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if resp == QMessageBox.Save:
                self.save(); QApplication.instance().quit()
            elif resp == QMessageBox.Discard:
                QApplication.instance().quit()
        else:
            QApplication.instance().quit()

    def closeEvent(self, event):
        event.ignore()
        self.close_app()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FaceBlurApp()
    w.show()
    sys.exit(app.exec())
