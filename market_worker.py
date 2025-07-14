import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from replica_screen import ReplicaScreen
import mss
import cv2
import pytesseract
import re
import os

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

    def analyze_candles_ocr(self, left_img, merged_left, right_img, merged_right):
        save_folder = "dummy"
        os.makedirs(save_folder, exist_ok=True)

        debug_3020 = left_img.copy()
        debug_1510 = right_img.copy()

        w_crop, h_crop = 130, 100
        found_3020 = None
        found_1510 = None

        #custom_config = r'--oem 3 --psm 3'
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=HL'

        # Process 3020 candles
        for idx, box in enumerate(merged_left):
            x1, y1, x2, y2 = box
            xc = (x1 + x2) // 2
            x_left = max(0, min(debug_3020.shape[1] - w_crop, xc - w_crop // 2))
            y_above = max(0, y1 - h_crop)
            y_below = min(debug_3020.shape[0] - h_crop, y2)

            # Draw rectangles for debug
            cv2.rectangle(debug_3020, (x_left, y_above), (x_left + w_crop, y_above + h_crop), (0, 255, 0), 2)
            cv2.rectangle(debug_3020, (x_left, y_below), (x_left + w_crop, y_below + h_crop), (0, 0, 255), 2)

            patch_above = left_img[y_above:y_above + h_crop, x_left:x_left + w_crop].copy()
            patch_below = left_img[y_below:y_below + h_crop, x_left:x_left + w_crop].copy()

            text_above = pytesseract.image_to_string(patch_above, config=custom_config).upper().strip()
            text_below = pytesseract.image_to_string(patch_below, config=custom_config).upper().strip()
            combined_text = text_above + " " + text_below

            print(f"3020 candle {idx + 1} - Extracted text above: '{text_above}'")
            print(f"3020 candle {idx + 1} - Extracted text below: '{text_below}'")

            # Debug save patches (optional)
            cv2.imwrite(f"{save_folder}/3020_candle{idx+1}_above.png", patch_above)
            cv2.imwrite(f"{save_folder}/3020_candle{idx+1}_below.png", patch_below)

            if re.search(r'\b(HH|LL)\b', combined_text):
                found_3020 = re.search(r'\b(HH|LL)\b', combined_text).group(0)
                print(f"3020 candle {idx + 1} OCR text: {found_3020}")
                break
            else:
                print(f"No HH or LL found in 3020 candle {idx + 1}")

        if not found_3020:
            print("No HH or LL found on 3020 side; skipping 1510 OCR.")
            cv2.imwrite(os.path.join(save_folder, "debug_3020.png"), debug_3020)
            cv2.imwrite(os.path.join(save_folder, "debug_1510.png"), debug_1510)
            return None

        # Process 1510 candles
        for idx, box in enumerate(merged_right):
            x1, y1, x2, y2 = box
            xc = (x1 + x2) // 2
            x_left = max(0, min(debug_1510.shape[1] - w_crop, xc - w_crop // 2))
            y_above = max(0, y1 - h_crop)
            y_below = min(debug_1510.shape[0] - h_crop, y2)

            cv2.rectangle(debug_1510, (x_left, y_above), (x_left + w_crop, y_above + h_crop), (0, 255, 0), 2)
            cv2.rectangle(debug_1510, (x_left, y_below), (x_left + w_crop, y_below + h_crop), (0, 0, 255), 2)

            patch_above = right_img[y_above:y_above + h_crop, x_left:x_left + w_crop].copy()
            patch_below = right_img[y_below:y_below + h_crop, x_left:x_left + w_crop].copy()

            text_above = pytesseract.image_to_string(patch_above, config=custom_config).upper().strip()
            text_below = pytesseract.image_to_string(patch_below, config=custom_config).upper().strip()
            combined_text = text_above + " " + text_below

            print(f"1510 candle {idx + 1} - Extracted text above: '{text_above}'")
            print(f"1510 candle {idx + 1} - Extracted text below: '{text_below}'")

            cv2.imwrite(f"{save_folder}/1510_candle{idx+1}_above.png", patch_above)
            cv2.imwrite(f"{save_folder}/1510_candle{idx+1}_below.png", patch_below)

            if re.search(r'\b(HL|LH)\b', combined_text):
                found_1510 = re.search(r'\b(HL|LH)\b', combined_text).group(0)
                print(f"1510 candle {idx + 1} OCR text: {found_1510}")
                break
            else:
                print(f"No HL or LH found in 1510 candle {idx + 1}")

        cv2.imwrite(os.path.join(save_folder, "debug_3020.png"), debug_3020)
        cv2.imwrite(os.path.join(save_folder, "debug_1510.png"), debug_1510)

        if found_3020 == "HH" and found_1510 == "HL":
            return "BUY"
        elif found_3020 == "LL" and found_1510 == "LH":
            return "SELL"
        else:
            return None

       
    def preprocess_for_ocr(patch):
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to enhance text visibility
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Optionally dilate or erode here if needed
        return thresh
    
    def crop(self,
            img: np.ndarray,
            box,                   # [x1, y1, x2, y2]
            width: int | None = None,
            height: int = 200,
            pad_x: int = 10,
            side: str = "above") -> np.ndarray:
        """
        Crop a patch directly above or below `box`.

        - If `width` is None, use candle width + 2*pad_x.
        - If the patch would exceed the image border, it is **clipped**
        (not shifted).
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box

        # --- horizontal limits 
        if width is None:
            width = (x2 - x1) + 2 * pad_x
        left = max(0, x1 - pad_x)
        right = min(w, left + width)
        left = max(0, right - width)       # recompute if we clipped on the right

        # --- vertical limits 
        if side == "above":
            top = max(0, y1 - height)
            bottom = y1
        else:  # below
            top = y2
            bottom = min(h, y2 + height)
            top = bottom - height          # recompute if we clipped at bottom

        return img[top:bottom, left:right].copy()


    def run(self):
        zoom = 0.6
        while self.running:
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

            m32 = lambda v: ((v + 31) // 32) * 32
            left_sz = (m32(left_monitor['width']), m32(left_monitor['height']))
            right_sz = (m32(right_monitor['width']), m32(right_monitor['height']))
        
            
            left_results = self.model.predict(
                source=left_img, verbose=False, stream=False,conf=0.01, iou=0.15, imgsz=(left_sz))
            right_results = self.model.predict(
                source=right_img,verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=(right_sz))

            left_boxes, left_scores = self.process_results(left_results)
            right_boxes, right_scores = self.process_results(right_results)

            keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
            merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])

            keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
            merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])

            decision = self.analyze_candles_ocr(left_img, merged_left, right_img, merged_right)
            if decision:
                print(decision)
            #draw coordinates
            left_img = self.draw_coords_only(left_img, merged_left)
            right_img = self.draw_coords_only(right_img, merged_right)

            # Emit images with coordinate overlays
            self.update_left.emit(left_img, merged_left)
            self.update_right.emit(right_img, merged_right)

            self.frame_count += 1
            print(f"\nFrame {self.frame_count} processed in {time.time() - start_time:.2f} sec.")

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
            # cv2.putText(img, top_left, (x1 - 40, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
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

        # self.left_replica = ReplicaScreen(
        #     0,
        #     400,
        #     650,
        #     1100,
        #     title="3020",
        #     trim_right=60
        # )

        # self.right_replica = ReplicaScreen(
        #     850,
        #     400,
        #     450,
        #     1100,
        #     title="1510",
        #     trim_right=0
        # )

        self.detection_thread = DetectionWorker(
            model=self.model,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames
        )

        # self.detection_thread.update_left.connect(self.left_replica.update_image_with_boxes)
        # self.detection_thread.update_right.connect(self.right_replica.update_image_with_boxes)
        self.detection_thread.finished.connect(self.on_finished)
        self.detection_thread.start()

    def on_finished(self):
        print("Detection finished.")
        self.app.quit()


if __name__ == "__main__":
    mw = MarketWorker()
    sys.exit(mw.app.exec_())