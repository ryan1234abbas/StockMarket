import sys
import time
import numpy as np
import pyautogui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from replica_screen import ReplicaScreen
import cv2
import mss
import numpy as np


class MarketWorker:
    def __init__(self):
        self.model = YOLO('/Users/koshabbas/Desktop/work/stock_market/runs/detect/train_185/weights/best.pt')
        self.app = QApplication.instance() or QApplication(sys.argv)

        # Region to capture — must match your replica screen's position and size exactly
        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410
        self.region = (self.offset_x, self.offset_y, self.width, self.height)

        # Create replica screen (display-only)
        self.replica = ReplicaScreen(self.offset_x, self.offset_y, self.width, self.height)

        self.frame_count = 0
        self.total_frames = 20 * 60 * 1  # e.g. 20 minutes at 1 fps

        # Start detection loop ASAP
        QTimer.singleShot(0, self.run_detection_loop)
        self.sct = mss.mss()

    def fast_screenshot(self):
        x, y, w, h = self.region
        monitor = {"top": y, "left": x, "width": w, "height": h}
        img = np.array(self.sct.grab(monitor))
        return img[..., :3]  # Remove alpha channel

    
    def run_detection_loop(self):
        if self.frame_count >= self.total_frames:
            print("Finished processing frames.")
            return

        start_time = time.time()

        # Fast screenshot using mss
        monitor = {
            "top": self.offset_y,
            "left": self.offset_x,
            "width": self.width,
            "height": self.height
        }
        img_np = np.array(self.sct.grab(monitor))[:, :, :3]  # Drop alpha channel

        # Run model prediction
        results = self.model.predict(source=img_np, conf=0.05, iou=0.15, imgsz=(self.width, self.height))

        boxes, scores = [], []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1) < 4 or (y2 - y1) < 20:
                    continue
                conf = box.conf[0].item()
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

        keep = self.non_max_suppression_fast(boxes, scores, iou_thresh=0.5)
        filtered_boxes = [boxes[i] for i in keep]
        merged_boxes = self.merge_vertically_close_boxes(filtered_boxes)

        # Update replica with new image and boxes
        self.replica.update_image_with_boxes(img_np, merged_boxes)

        self.frame_count += 1
        elapsed = time.time() - start_time
        print(f"Frame {self.frame_count}/{self.total_frames} processed in {elapsed:.2f} seconds.")

        QTimer.singleShot(0, self.run_detection_loop)


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

if __name__ == "__main__":
    mw = MarketWorker()
    sys.exit(mw.app.exec_())



'''Use for overlaying models prediction on live screen'''
    # def run(self, duration_minutes=20, fps=1):
    #     print("Started screenshotting and detection loop.")
    #     total_frames = duration_minutes * 60 * fps
    #     frame_count = 0

    #     try:
    #         while frame_count < total_frames:
    #             start_time = time.time()

    #             all_boxes = []
    #             for tab in ["3020", "1510"]:
    #                 img = self.tab_screenshotter.capture_screenshot(tab)
    #                 if img is None:
    #                     print(f"Could not capture screenshot for tab {tab}")
    #                     continue

    #                 img_np = np.array(img.convert("RGB"))
    #                 original_w, original_h = img.size

    #                 results = self.model.predict(source=img_np, conf=0.2, iou=0.1)
                    
    #                 boxes, scores = [], []
    #                 for result in results:
    #                     # YOLO’s resized input size for this image (width, height)
    #                     pred_w, pred_h = result.orig_shape[1], result.orig_shape[0]

    #                     for box in result.boxes:
    #                         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #                         conf = box.conf[0].item()

    #                         # Filter out very small boxes
    #                         if (x2 - x1) < 4 or (y2 - y1) < 20:
    #                             continue

    #                         # Rescale coords from resized image to original screenshot size
    #                         x1 = x1 * original_w / pred_w
    #                         x2 = x2 * original_w / pred_w
    #                         y1 = y1 * original_h / pred_h
    #                         y2 = y2 * original_h / pred_h

    #                         boxes.append([int(x1), int(y1), int(x2), int(y2)])
    #                         scores.append(conf)

    #                 keep = self.non_max_suppression_fast(boxes, scores, iou_thresh=0.3)
    #                 filtered_boxes = [boxes[i] for i in keep]

    #                 merged_boxes = self.merge_vertically_close_boxes(filtered_boxes)
                  
    #                 offset_x, offset_y = self.tab_screenshotter.get_tab_offset(tab)  # you implement this
    #                 # Adjust boxes to be relative to full screen
    #                 adjusted_boxes = []
    #                 for x1, y1, x2, y2 in merged_boxes:
    #                     adjusted_boxes.append([
    #                         x1 + offset_x,
    #                         y1 + offset_y,
    #                         x2 + offset_x,
    #                         y2 + offset_y,
    #                     ])

    #                 all_boxes.extend(merged_boxes)

    #                 print(f"[{tab}] Frame {frame_count + 1} | Raw: {len(boxes)}, Final: {len(merged_boxes)}")

    #             self.overlay.update_boxes(all_boxes)
    #             self.overlay.raise_()  # Keeps overlay above other windows

    #             frame_count += 1
    #             time.sleep(max(0, (1.0 / fps) - (time.time() - start_time)))

    #     except KeyboardInterrupt:
    #         print("Stopped by user.")



'''Use for detecting each frame'''
    # def run(self):
    #     print("Started screenshotting and detection loop.")

    #     cv2.namedWindow("3020 Detection", cv2.WINDOW_NORMAL)
    #     cv2.namedWindow("1510 Detection", cv2.WINDOW_NORMAL)

    #     while True:
    #         try:
    #             # Capture screenshots and save them in the script's directory
    #             self.tab_screenshotter.capture_screenshot("3020")
    #             self.tab_screenshotter.capture_screenshot("1510")

    #             # Construct full paths to the saved screenshots
    #             img_path_3020 = os.path.join(self.tab_screenshotter.save_dir, "3020.png")
    #             img_path_1510 = os.path.join(self.tab_screenshotter.save_dir, "1510.png")

    #             # Run detection and show result for tab 3020
    #             self.detect_candles(self.model, img_path_3020)
    #             img_3020_out = cv2.imread(img_path_3020)
    #             if img_3020_out is not None:
    #                 cv2.imshow("3020 Detection", img_3020_out)
    #             else:
    #                 print("Failed to read 3020.png after detection.")

    #             # Run detection and show result for tab 1510
    #             self.detect_candles(self.model, img_path_1510)
    #             img_1510_out = cv2.imread(img_path_1510)
    #             if img_1510_out is not None:
    #                 cv2.imshow("1510 Detection", img_1510_out)
    #             else:
    #                 print("Failed to read 1510.png after detection.")

    #             # Exit loop on 'q' key press
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 print("Exiting on user request.")
    #                 break

    #         except KeyboardInterrupt:
    #             print("Exiting MarketWorker loop on user interrupt.")
    #             break

    #     cv2.destroyAllWindows()

