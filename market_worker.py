import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from replica_screen import ReplicaScreen
import mss
import cv2
class DetectionWorker(QThread):
    update_left = pyqtSignal(np.ndarray, list)
    update_right = pyqtSignal(np.ndarray, list)
    finished = pyqtSignal()

    def __init__(self, model, offset_x, offset_y, width, height, total_frames):
        super().__init__()
        self.model = model
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.frame_count = 0
        self.sct = mss.mss()
        self.running = True

    def run(self):
        zoom = 0.6
        while self.running and self.frame_count < self.total_frames:
            start_time = time.time()

            # LEFT REGION
            lw_orig = self.width // 2
            lh_orig = self.height
            lw_zoom = int(lw_orig / zoom)
            lh_zoom = int(lh_orig / zoom)
            left_monitor = {
                "top": self.offset_y - (lh_zoom - lh_orig) // 2,
                "left": (self.offset_x - (lw_zoom - lw_orig) // 2) + 20,
                "width": lw_zoom - 70,
                "height": lh_zoom
            }

            # RIGHT REGION
            rw_orig = self.width - lw_orig
            rh_orig = self.height
            rw_zoom = int(rw_orig / zoom)
            rh_zoom = int(rh_orig / zoom)
            right_monitor = {
                "top": self.offset_y - (rh_zoom - rh_orig) // 2,
                "left": (self.offset_x + lw_orig - (rw_zoom - rw_orig) // 2) + 180,
                "width": rw_zoom - 100,
                "height": rh_zoom
            }

            left_img = np.array(self.sct.grab(left_monitor))[:, :, :3]
            right_img = np.array(self.sct.grab(right_monitor))[:, :, :3]

            left_results = self.model.predict(
                source=left_img, conf=0.01, iou=0.15, imgsz=(left_monitor['width'], left_monitor['height']))
            right_results = self.model.predict(
                source=right_img, conf=0.01, iou=0.15, imgsz=(right_monitor['width'], right_monitor['height']))

            left_boxes, left_scores = self.process_results(left_results)
            right_boxes, right_scores = self.process_results(right_results)

            keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
            merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])

            keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
            merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])
            
            #print each box's top left (x0,y0) coordinates
            
            #3020 
            top_left_coords_3020 = sorted([(box[0], box[1]) for box in merged_left], key=lambda c: c[0])
            print("3020:", top_left_coords_3020)

            #1510 
            top_left_coords_1510 = sorted([(box[0], box[1]) for box in merged_right], key=lambda c: c[0])
            print("1510:", top_left_coords_1510)

            self.update_left.emit(left_img, merged_left)
            self.update_right.emit(right_img, merged_right)
            
            #draw coordinates
            left_img = self.draw_coords_only(left_img, merged_left)
            right_img = self.draw_coords_only(right_img, merged_right)

            # Emit images with coordinate overlays
            self.update_left.emit(left_img, merged_left)
            self.update_right.emit(right_img, merged_right)

            self.frame_count += 1
            print(f"Frame {self.frame_count}/{self.total_frames} processed in {time.time() - start_time:.2f} sec.")

        self.finished.emit()

    def process_results(self, results):
        boxes = []
        scores = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1) < 4 or (y2 - y1) < 20:
                    continue
                boxes.append([x1, y1, x2, y2])
                scores.append(box.conf[0].item())
        #print("Detected boxes:", boxes)
        return boxes, scores
    
    def draw_coords_only(self, img, boxes):
        img = img.astype(np.uint8).copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            # Text for each corner
            top_left = f"({x1},{y1})"
            top_right = f"({x2},{y1})"
            bottom_left = f"({x1},{y2})"
            bottom_right = f"({x2},{y2})"

            # Put text near each corner (adjust offsets so text doesn't overlap box edges)
            cv2.putText(img, top_left, (x1 - 40, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # cv2.putText(img, top_right, (x2 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # cv2.putText(img, bottom_left, (x1 - 40, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # cv2.putText(img, bottom_right, (x2 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return img


    def non_max_suppression_fast(self, boxes, scores, iou_thresh=0.4):
        if not boxes:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def merge_vertically_close_boxes(self, boxes, y_thresh=30, x_thresh=15):
        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            x1a, y1a, x2a, y2a = box1
            group = [box1]
            for j, box2 in enumerate(boxes):
                if j <= i or j in used:
                    continue
                x1b, y1b, x2b, y2b = box2
                if abs(x1a - x1b) < x_thresh and abs(x2a - x2b) < x_thresh:
                    if abs(y1a - y2b) < y_thresh or abs(y2a - y1b) < y_thresh:
                        group.append(box2)
                        used.add(j)

            xs = [b[0] for b in group] + [b[2] for b in group]
            ys = [b[1] for b in group] + [b[3] for b in group]
            merged.append([min(xs), min(ys), max(xs), max(ys)])
            used.add(i)

        return merged


class MarketWorker:
    def __init__(self):
        self.model = YOLO('/Users/koshabbas/Desktop/work/stock_market/runs/detect/train_19/weights/last.pt')
        self.app = QApplication.instance() or QApplication(sys.argv)

        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410

        self.total_frames = 20 * 60 * 1  # 20 minutes at 1 fps

        self.left_replica = ReplicaScreen(
            0,
            400,
            650,
            1100,
            title="3020",
            trim_right=60
        )

        self.right_replica = ReplicaScreen(
            850,
            400,
            450,
            1100,
            title="1510",
            trim_right=0
        )

        self.detection_thread = DetectionWorker(
            model=self.model,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames
        )

        self.detection_thread.update_left.connect(self.left_replica.update_image_with_boxes)
        self.detection_thread.update_right.connect(self.right_replica.update_image_with_boxes)
        self.detection_thread.finished.connect(self.on_finished)
        self.detection_thread.start()

    def on_finished(self):
        print("Detection finished.")
        self.app.quit()


if __name__ == "__main__":
    mw = MarketWorker()
    sys.exit(mw.app.exec_())
