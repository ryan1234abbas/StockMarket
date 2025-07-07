from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.boxes = []
        self.init_ui()

    def init_ui(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus)

        # 👇 Force it to stay above other windows repeatedly
        self.setGeometry(0, 0, self.screen().size().width(), self.screen().size().height())
        self.show()
        self.raise_()  # <-- Push to front


    def update_boxes(self, boxes):
        self.boxes = boxes
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # ✅ Fully transparent background
        painter.fillRect(self.rect(), Qt.transparent)

        # ✅ Green outline for boxes
        painter.setPen(QPen(QColor(0, 255, 0, 200), 2))
        painter.setBrush(Qt.NoBrush)

        for x1, y1, x2, y2 in self.boxes:
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

