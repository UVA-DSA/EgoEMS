#!/usr/bin/env python3
import sys
import json
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QDialog, QLabel,
    QInputDialog
)
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


class PointSelector(QDialog):
    """Dialog that lets the user click multiple points on a frame."""
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Click to add points; press Done when finished")
        rgb = frame[..., ::-1]
        h, w, _ = rgb.shape
        bpl = 3 * w
        img = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        self.pix = QPixmap.fromImage(img)
        self.label = QLabel(self)
        self.label.setPixmap(self.pix)
        self.label.setFixedSize(w, h)
        self.label.mousePressEvent = self.on_click

        self.points = []  # list of (x,y) tuples

        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(done_btn)

    def on_click(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        pos = ev.pos()
        x, y = pos.x(), pos.y()
        self.points.append((x, y))

        # draw a small circle at the click
        painter = QPainter(self.pix)
        pen = QPen(Qt.red)
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawEllipse(pos, 5, 5)
        painter.end()
        self.label.setPixmap(self.pix)


class ROIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EgoEMS - Manual Censoring Tool")
        self.resize(800, 600)

        self.cap = None
        self.num_frames = 0
        self.fps = 30
        self.current = 0
        self.annotations = []  # list of {"frame":int,"id":str,"points":[[x,y],...]}

        # Video display
        self.video_label = QLabel(alignment=Qt.AlignCenter)

        # Controls
        load_btn  = QPushButton("Load Video");   load_btn.clicked.connect(self.load_video)
        prev_btn  = QPushButton("⯇ Prev");       prev_btn.clicked.connect(self.prev_frame)
        self.play_btn = QPushButton("Play");     self.play_btn.clicked.connect(self.toggle_play)
        next_btn  = QPushButton("Next ⯈");       next_btn.clicked.connect(self.next_frame)
        mark_btn  = QPushButton("Mark Points");  mark_btn.clicked.connect(self.mark_roi)
        sam_btn  = QPushButton("SAM2 Engine");  sam_btn.clicked.connect(self.process_sam)
        close_btn = QPushButton("Close Window"); close_btn.clicked.connect(self.save_and_close)

        ctrl = QHBoxLayout()
        for btn in (load_btn, prev_btn, self.play_btn, next_btn, mark_btn,sam_btn, close_btn):
            ctrl.addWidget(btn)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(self.video_label, 1)
        lay.addLayout(ctrl)
        self.setCentralWidget(root)

        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video.")
            return
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current = 0
        self.annotations.clear()
        self.show_frame(0)

    def show_frame(self, idx):
        idx = max(0, min(idx, self.num_frames - 1))
        self.current = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = frame[..., ::-1]
        h, w, _ = rgb.shape
        bpl = 3 * w
        img = QImage(rgb.data.tobytes(), w, h, bpl, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def process_sam(self):
        QMessageBox.information(self, "Info", "SAM2 Engine processing is not implemented yet.")

    def prev_frame(self):
        if not self.timer.isActive():
            self.show_frame(self.current - 1)

    def next_frame(self):
        if self.timer.isActive() or self.sender() is self.timer:
            if self.current < self.num_frames - 1:
                self.show_frame(self.current + 1)
            else:
                self.toggle_play()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            interval = int(1000 / self.fps)
            self.timer.start(interval)
            self.play_btn.setText("Pause")

    def mark_roi(self):
        # grab current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current)
        ret, frame = self.cap.read()
        if not ret:
            return

        # open point selector dialog
        dlg = PointSelector(frame, self)
        if dlg.exec_() == QDialog.Accepted and dlg.points:
            oid, ok = QInputDialog.getText(self, "Annotation ID", "Enter object ID:")
            if ok and oid:
                self.annotations.append({
                    "frame": self.current,
                    "id": oid,
                    "points": dlg.points
                })
                QMessageBox.information(
                    self, "Marked",
                    f"Frame {self.current}: ID={oid}, points={dlg.points}"
                )

    def save_and_close(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", "annotations.json", "JSON Files (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.annotations, f, indent=2)
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ROIApp()
    w.show()
    sys.exit(app.exec())
