#!/usr/bin/env python3
import sys, cv2, av, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QSlider, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QRubberBand, QWidget,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer, QRect, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap

# Constants
TIME_TO_TRACK = 2 # seconds to track after marking ROI

class ROISelector(QDialog):
    # This class is unchanged and correct.
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
    # All methods except 'save' are unchanged from the previous correct version
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EgoEMS - Face Deidentification Verification & Correction Tool")
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
        self.track_btn  = QPushButton("Track Face & Blur");self.track_btn.clicked.connect(self.track_blur)
        self.save_btn   = QPushButton("Save Video"); self.save_btn.clicked.connect(self.save)
        self.close_btn  = QPushButton("Close");      self.close_btn.clicked.connect(self.close_app)
        ctrl = QHBoxLayout()
        for w in (load_btn, self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn, self.mark_btn, self.track_btn, self.save_btn, self.close_btn):
            ctrl.addWidget(w)
        root = QWidget(); lay = QVBoxLayout(root); lay.addWidget(self.video_label, 1); lay.addWidget(self.slider); lay.addLayout(ctrl)
        self.setCentralWidget(root)
        self.timer = QTimer(self); self.timer.timeout.connect(self.next_frame)
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn, self.mark_btn, self.track_btn, self.save_btn, self.close_btn):
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
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn, self.mark_btn, self.track_btn, self.save_btn, self.close_btn):
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
        rgb = frame[..., ::-1]; h, w, _ = rgb.shape; bpl = 3*w
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

    def track_blur(self):
        if not self.rois: QMessageBox.warning(self, "No ROI", "Please mark at least one ROI first."); return
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn, self.mark_btn, self.track_btn):
            btn.setEnabled(False)
        QMessageBox.information(self, "Tracking", "Running tracker…"); QApplication.processEvents()
        to_track = int(TIME_TO_TRACK * self.fps)
        start_frames = [sf for sf, _ in self.rois]
        min_sf = min(start_frames); max_ef = max(sf + to_track for sf, _ in self.rois)
        max_end_frame = min(max_ef, self.total - 1)
        cap2 = cv2.VideoCapture(self.video_path); cap2.set(cv2.CAP_PROP_POS_FRAMES, min_sf)
        active_trackers = []
        for frame_idx in range(min_sf, max_end_frame + 1):
            ret, frame = cap2.read()
            if not ret: break
            for start_frame, bbox in self.rois:
                if start_frame == frame_idx:
                    tracker = cv2.TrackerCSRT_create(); tracker.init(frame, bbox)
                    active_trackers.append({'tracker': tracker, 'end_frame': start_frame + to_track})
            for active in active_trackers:
                if frame_idx < active['end_frame']:
                    ok, box = active['tracker'].update(frame)
                    if ok:
                        x, y, w, h = map(int, box)
                        x1, y1 = max(0, x), max(0, y); x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2]
                            blur_w = (w//4)*2 + 1; blur_h = (h//4)*2 + 1
                            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_w, blur_h), 0)
            self.blurred_frames[frame_idx] = frame.copy()
            if frame_idx % 10 == 0: self._display(frame); QApplication.processEvents()
        cap2.release()
        next_frame_to_show = min(max_end_frame + 1, self.total - 1)
        self.show_frame(next_frame_to_show)
        # self.show_frame(self.current)
        QMessageBox.information(self, "Completed", "All ROIs tracked and blurred.")
        self.rois.clear(); self.saved = False
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn, self.mark_btn, self.track_btn, self.save_btn):
            btn.setEnabled(True)
    
    # ==============================================================================
    # ─── THE FINAL, WORKING SAVE METHOD ───────────────────────────────────────────
    # ==============================================================================
    def save(self):
        if not self.blurred_frames:
            QMessageBox.warning(self, "Nothing to Save", "No blurred frames have been generated.")
            return

        orig_dir = os.path.dirname(self.video_path)
        default_fname = f"Verified_{self.video_name}.mp4"
        default_path = os.path.join(orig_dir, default_fname)
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", default_path, "MP4 Files (*.mp4)")
        if not path: return

        in_container = None
        out_container = None
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
                            # === THE DEFINITIVE FIX ===
                            # Create a new frame from our blurred NumPy array.
                            new_frame = av.VideoFrame.from_ndarray(numpy_blurred_frame, format='bgr24')
                            # Copy the Presentation Timestamp AND the Time Base from the original frame.
                            # This is the critical step that ensures the frame is encoded at the correct time.
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
            
            progress.close()

            if was_cancelled:
                QMessageBox.information(self, "Cancelled", "Save operation was cancelled.")
            else:
                self.saved = True
                QMessageBox.information(self, "Saved", f"Video with audio successfully written to\n{path}")

        except av.AVError as e:
            QMessageBox.critical(self, "Error", f"An error occurred during saving: {e}")
        finally:
            # Robustly close containers and clean up failed files
            if in_container: in_container.close()
            if out_container: out_container.close()
            if 'was_cancelled' in locals() and was_cancelled and os.path.exists(path):
                os.remove(path)

    def close_app(self):
        if self.blurred_frames and not self.saved:
            resp = QMessageBox.question(self, "Unsaved Changes", "Save before closing?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if resp == QMessageBox.Save: self.save(); QApplication.instance().quit()
            elif resp == QMessageBox.Discard: QApplication.instance().quit()
        else: QApplication.instance().quit()

    def closeEvent(self, event):
        event.ignore()
        self.close_app()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FaceBlurApp()
    w.show()
    sys.exit(app.exec())