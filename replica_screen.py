import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtCore import Qt

class ReplicaScreen(QWidget):
    def __init__(self, offset_x, offset_y, width, height):
        super().__init__()
        self.region = (offset_x, offset_y, width, height)
        self.boxes = []
        self.label = QLabel(self)
        self.setGeometry(offset_x, offset_y, width, height)
        self.setWindowTitle("Replica Screen")
        self.show()

    def update_image_with_boxes(self, img_np, boxes):
        h, w, _ = img_np.shape
        # Create QImage from numpy array
        qimg = QImage(img_np.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 0, 200), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        for box in boxes:
            x1, y1, x2, y2 = box
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        painter.end()

        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.size())
        self.update()
