import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtCore import Qt
import cv2
class ReplicaScreen(QWidget):
    def __init__(self, offset_x, offset_y, width, height, title="Replica Screen", trim_right=0):
        super().__init__()
        self.region = (offset_x, offset_y, width, height)
        self.trim_right = trim_right
        
        self.boxes = []
        self.label = QLabel(self)
        self.setWindowTitle(title)
        self.resize(width, height)
        self.move(offset_x, offset_y) 
        
        # self.label.setFixedSize(width, height)
        # self.setFixedSize(width, height)

        self.show()


    def update_image_with_boxes(self, img_np, boxes):
        if self.trim_right > 0:
            img_np = img_np[:, :-self.trim_right, :]
            
        trim_top = 150 
        img_np = img_np[trim_top:, :, :]  

        rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scale_factor = 0.4
        scaled_w, scaled_h = int(w * scale_factor), int(h * scale_factor)
        scaled_pixmap = pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio)

        painter = QPainter(scaled_pixmap)
        pen = QPen(QColor(0, 255, 0, 200), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_factor)
            y1 = int((y1 - trim_top) * scale_factor)  
            x2 = int(x2 * scale_factor)
            y2 = int((y2 - trim_top) * scale_factor)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        painter.end()

        self.label.setPixmap(scaled_pixmap)
        self.label.resize(scaled_pixmap.size())
        self.resize(scaled_pixmap.size())
        
        self.label.repaint()
        self.repaint()
        self.update()
