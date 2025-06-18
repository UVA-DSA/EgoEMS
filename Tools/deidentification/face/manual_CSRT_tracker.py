#!/usr/bin/env python3
import sys, cv2, av
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QSlider, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QRubberBand, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QRect, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap
from fractions import Fraction
# Constants
TIME_TO_TRACK = 5  # seconds to track after marking ROI

# ── ROI DRAWER ────────────────────────────────────────────────────────────────
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

# ── MAIN APP ─────────────────────────────────────────────────────────────────
class FaceBlurApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EgoEMS - Face Deidentification Verification & Correction Tool")
        self.resize(800, 600)

        # Video state
        self.cap = None
        self.total = 0
        self.fps = 30
        self.current = 0
        self.video_path = ""

        # ROI & tracking state
        # Changed self.bbox = None to self.rois = []
        self.rois = []
        self.start_frame = None
        self.blurred_frames = {}  # frame_idx → blurred_frame
        self.saved = False

        # UI
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)

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
        for w in (load_btn, self.play_btn,
                  self.prev3_btn, self.prev_btn, self.next_btn, self.next3_btn,
                  self.mark_btn, self.track_btn, self.save_btn, self.close_btn):
            ctrl.addWidget(w)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(self.video_label, 1)
        lay.addWidget(self.slider)
        lay.addLayout(ctrl)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # Disable until load
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn,
                    self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn,
                    self.save_btn, self.close_btn):
            btn.setEnabled(False)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video")
            return
        self.cap = cap
        self.video_path = path
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.slider.setRange(0, self.total-1)
        self.slider.setEnabled(True)
        self.current = 0
        self.bbox = None
        self.start_frame = None
        self.blurred_frames.clear()
        self.saved = False

        # Enable controls
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn,
                    self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn,
                    self.save_btn, self.close_btn):
            btn.setEnabled(True)
        self.show_frame(0)

    def show_frame(self, idx):
        idx = max(0, min(self.total-1, idx))
        self.current = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self._display(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _display(self, frame):
        rgb = frame[..., ::-1]
        h, w, _ = rgb.shape
        bpl = 3*w
        qimg = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def seek(self, pos):
        if self.timer.isActive():
            self.play()
        self.show_frame(pos)

    def play(self):
        if not self.cap:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            interval = int(1000/self.fps)
            self.timer.start(interval)
            self.play_btn.setText("Pause")

    def next_frame(self):
        if self.current < self.total-1:
            self.show_frame(self.current+1)
        else:
            self.play()

    def mark_roi(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current)
        ret, frame = self.cap.read()
        if not ret: return
        dlg = ROISelector(frame, self)
        if dlg.exec_() == QDialog.Accepted and dlg.rect:
            bbox = tuple(dlg.rect)
            self.start_frame = self.current
            ## added the line bellow
            self.rois.append((self.start_frame, bbox))
            QMessageBox.information(self, "ROI",
                f"Start frame {self.start_frame}, BBox {self.bbox}")

    def track_blur(self):
        if len(self.rois) <1:
            QMessageBox.warning(self, "No ROI", "Please mark at least one ROI first.")
            return

        for btn in (self.play_btn, self.prev3_btn, self.prev_btn,
                    self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn):
            btn.setEnabled(False)
        QMessageBox.information(self, "Tracking", "Running tracker…")

        #caluclates frame window
        to_track = int(TIME_TO_TRACK * self.fps)
        start_frames = [sf for sf, _ in self.rois]
        min_sf = min(start_frames)
        max_end = max(start_frames) + to_track

        cap2 = cv2.VideoCapture(self.video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, min_sf)
        current = min_sf

        # current (start, tracker) pairs running in current pass
        active = []

        # useses caluted end frame to pass through once
        while current <= max_end:
            ret, frame = cap2.read()
            if not ret:
                break

            for sf, bbox in self.rois:
                if sf == current:
                    tr = cv2.TrackerCSRT_create()
                    tr.init(frame, bbox)
                    active.append((sf, tr))

            #update & blur all active trackers
            for sf, tr in active:
                rel = current - sf
                if 0 <= rel < to_track:
                    ok, box = tr.update(frame)
                    if ok:
                        x, y, w, h = map(int, box)
                        x, y = max(0, x), max(0, y)
                        w = min(w, frame.shape[1]-x)
                        h = min(h, frame.shape[0]-y)
                        if w > 0 and h > 0:
                            roi = frame[y:y+h, x:x+w]
                            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51,51), 0)

            self.blurred_frames[current] = frame.copy()
            self._display(frame)
            QApplication.processEvents()
            current += 1

        cap2.release()

        self.show_frame(self.current)
        QMessageBox.information(self, "Completed", "All ROIs tracked and blurred.")
        # clears self.rois
        self.rois.clear()
        for btn in (self.play_btn, self.prev3_btn, self.prev_btn,
                    self.next_btn, self.next3_btn,
                    self.mark_btn, self.track_btn, self.save_btn):
            btn.setEnabled(True)

    def save(self):
    # 1) Nothing to do?
        if not self.blurred_frames:
            QMessageBox.warning(self, "Nothing", "No blurred segments to save.")
            return

        # 2) Ask where to save
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "blurred_with_audio.mp4", "MP4 Files (*.mp4)"
        )
        if not path:
            return

        # 3) Open input container & find streams
        in_cont = av.open(self.video_path)
        in_v    = in_cont.streams.video[0]
        in_a    = next((s for s in in_cont.streams if s.type == "audio"), None)

        # 4) Compute a true 1/fps time_base
        fps = in_v.average_rate            # Fraction(30,1), etc.
        tb  = Fraction(fps.denominator, fps.numerator)

        # 5) Open output container
        out_cont = av.open(path, mode="w")

        # 6) Add a libx264 video stream in lossless mode
        out_v = out_cont.add_stream("libx264", rate=fps)
        ctx   = out_v.codec_context

        #  6a) Tell the encoder how to timestamp and size frames
        ctx.time_base = tb
        out_v.time_base = tb

        #  6b) Match the input resolution & force yuv420p
        ctx.width   = in_v.codec_context.width
        ctx.height  = in_v.codec_context.height
        ctx.pix_fmt = "yuv420p"
        out_v.width  = ctx.width
        out_v.height = ctx.height
        out_v.pix_fmt = ctx.pix_fmt

        #  6c) Lossless: CRF=0, you can raise CRF if you want smaller size
        ctx.options = {"crf": "0", "preset": "slow"}

        # 7) Configure audio passthrough, if present
        if in_a:
            out_a = out_cont.add_stream(in_a.codec_context.name,
                                        rate=in_a.codec_context.sample_rate)
            ac    = out_a.codec_context
            ac.channels    = in_a.codec_context.channels
            ac.layout      = in_a.codec_context.layout
            ac.sample_rate = in_a.codec_context.sample_rate

        # 8) Read, blur-substitute, reformat, timestamp, encode + mux each frame
        cap   = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            # pick either original or blurred
            frm = self.blurred_frames.get(i, frame)

            # wrap & convert into encoder’s expected format/size
            vf = (
                av.VideoFrame.from_ndarray(frm, format="bgr24")
                .reformat(format=ctx.pix_fmt, width=ctx.width, height=ctx.height)
            )

            # stamp with proper PTS in 1/fps timebase
            vf.pts       = i
            vf.time_base = tb

            for pkt in out_v.encode(vf):
                out_cont.mux(pkt)

        # 9) Flush the video encoder
        for pkt in out_v.encode(None):
            out_cont.mux(pkt)
        cap.release()

        # 10) Mux audio unchanged
        if in_a:
            for packet in in_cont.demux(in_a):
                packet.stream = out_a
                out_cont.mux(packet)

        # 11) Close containers & notify
        out_cont.close()
        in_cont.close()
        self.saved = True
        QMessageBox.information(
            self, "Saved", f"Video (with audio) written to {path}"
        )

    def close_app(self):
        if self.blurred_frames and not self.saved:
            resp = QMessageBox.question(
                self, "Unsaved", "You have unsaved blurred segments. Save now?"
            )
            if resp == QMessageBox.Yes:
                self.save()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FaceBlurApp()
    w.show()
    sys.exit(app.exec())
