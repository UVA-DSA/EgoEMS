#!/usr/bin/env python3
import sys, cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QSlider, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QRubberBand, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QRect, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap

# Constants
TIME_TO_TRACK = 5.0  # seconds to track and blur

# ── ROI DRAWER ────────────────────────────────────────────────────────────────
class ROISelector(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw ROI")
        rgb = frame[..., ::-1]
        h, w, _ = rgb.shape
        bpl = 3*w
        qimg = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        lbl  = QLabel(self)
        lbl.setPixmap(pix)
        lbl.setFixedSize(w, h)

        self.band   = QRubberBand(QRubberBand.Rectangle, lbl)
        self.origin = QPoint()
        self.rect   = None

        lbl.mousePressEvent   = self.onPress
        lbl.mouseMoveEvent    = self.onMove
        lbl.mouseReleaseEvent = self.onRelease

        layout = QVBoxLayout(self)
        layout.addWidget(lbl)

    def onPress(self, ev):
        if ev.button()==Qt.LeftButton:
            self.origin = ev.pos()
            self.band.setGeometry(QRect(self.origin, QSize()))
            self.band.show()
    def onMove(self, ev):
        if not self.origin.isNull():
            self.band.setGeometry(QRect(self.origin, ev.pos()).normalized())
    def onRelease(self, ev):
        if ev.button()==Qt.LeftButton:
            self.band.hide()
            r = self.band.geometry()
            self.rect = (r.x(), r.y(), r.width(), r.height())
            self.accept()

# ── MAIN APP ─────────────────────────────────────────────────────────────────
class FaceBlurApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Blur Segmenter")
        self.resize(800, 600)

        # video state
        self.cap = None
        self.total = 0
        self.fps = 30
        self.current = 0
        self.video_path = ""

        # ROI & tracking state
        self.bbox = None
        self.start_frame = None
        self.blurred_frames = {}   # frame_idx -> blurred_frame
        self.saved = False

        # UI
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)

        load_btn  = QPushButton("Load Video");    load_btn.clicked.connect(self.load)
        self.play_btn  = QPushButton("Play");     self.play_btn.clicked.connect(self.play)
        self.mark_btn  = QPushButton("Mark ROI"); self.mark_btn.clicked.connect(self.mark_roi)
        self.track_btn = QPushButton("Track & Blur"); self.track_btn.clicked.connect(self.track_blur)
        self.save_btn  = QPushButton("Save Video");   self.save_btn.clicked.connect(self.save)
        self.close_btn = QPushButton("Close");        self.close_btn.clicked.connect(self.close_app)

        ctrl = QHBoxLayout()
        for b in (load_btn, self.play_btn, self.mark_btn, self.track_btn, self.save_btn, self.close_btn):
            ctrl.addWidget(b)

        root = QWidget()
        lay  = QVBoxLayout(root)
        lay.addWidget(self.video_label, 1)
        lay.addWidget(self.slider)
        lay.addLayout(ctrl)
        self.setCentralWidget(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # disable until loaded
        for w in (self.play_btn, self.mark_btn, self.track_btn, self.save_btn, self.close_btn):
            w.setEnabled(False)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path: return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video")
            return
        self.cap = cap
        self.video_path = path
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.slider.setRange(0, self.total-1)
        self.slider.setEnabled(True)
        self.current = 0
        self.bbox = None
        self.start_frame = None
        self.blurred_frames.clear()
        self.saved = False

        # enable after load
        for w in (self.play_btn, self.mark_btn, self.track_btn, self.close_btn):
            w.setEnabled(True)
        self.save_btn.setEnabled(False)

        self.show_frame(0)

    def show_frame(self, idx):
        idx = max(0, min(self.total-1, idx))
        self.current = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret: return
        self._display(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _display(self, frame):
        rgb = frame[..., ::-1]
        h, w, _ = rgb.shape
        bpl = 3*w
        qimg = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def seek(self, pos):
        if self.timer.isActive():
            self.play()
        self.show_frame(pos)

    def play(self):
        if not self.cap: return
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
            self.bbox = tuple(dlg.rect)
            self.start_frame = self.current
            QMessageBox.information(self, "ROI", f"Start {self.start_frame}, BBox {self.bbox}")

    def track_blur(self):
        if not self.bbox or self.start_frame is None:
            QMessageBox.warning(self, "No ROI", "Please mark ROI first.")
            return

        # disable during tracking
        for w in (self.play_btn, self.mark_btn, self.track_btn):
            w.setEnabled(False)
        QMessageBox.information(self, "Tracking", "Running tracker…")

        cap2 = cv2.VideoCapture(self.video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, first = cap2.read()
        tracker = cv2.TrackerCSRT_create()
        tracker.init(first, self.bbox)

        maxf = int(TIME_TO_TRACK * self.fps)
        for i in range(maxf):
            ret, frm = cap2.read()
            if not ret: break
            ok, box = tracker.update(frm)
            if not ok: break
            x,y,w,h = map(int, box)
            x,y = max(0,x), max(0,y)
            w = min(w, frm.shape[1]-x)
            h = min(h, frm.shape[0]-y)
            if w>0 and h>0:
                roi = frm[y:y+h, x:x+w]
                frm[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51,51), 0)
            idx = self.start_frame + i
            self.blurred_frames[idx] = frm.copy()
            self._display(frm)
            QApplication.processEvents()

        self.end_frame = idx
        self.show_frame(self.end_frame)
        cap2.release()

        QMessageBox.information(self, "Done",
            f"Segment blurred {self.start_frame}→{self.end_frame}")

        # re-enable
        for w in (self.play_btn, self.mark_btn, self.track_btn, self.save_btn):
            w.setEnabled(True)

    def save(self):
        if not self.blurred_frames:
            QMessageBox.warning(self, "Nothing", "No blurred segments to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "blurred_output.mp4", "MP4 Files (*.mp4)")
        if not path: return

        cap2 = cv2.VideoCapture(self.video_path)
        total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap2.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc= cv2.VideoWriter_fourcc(*"mp4v")
        out   = cv2.VideoWriter(path, fourcc, fps, (w,h))

        for i in range(total):
            if i in self.blurred_frames:
                out.write(self.blurred_frames[i])
            else:
                ret, fr = cap2.read()
                if not ret: break
                out.write(fr)
        cap2.release()
        out.release()
        self.saved = True
        QMessageBox.information(self, "Saved", f"Video written to {path}")

    def close_app(self):
        if self.blurred_frames and not self.saved:
            resp = QMessageBox.question(self, "Unsaved", "You have unsaved segments. Save now?")
            if resp == QMessageBox.Yes:
                self.save()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FaceBlurApp()
    w.show()
    sys.exit(app.exec())
