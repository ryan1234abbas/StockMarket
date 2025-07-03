import time
import cv2
import numpy as np
from PIL import Image
from take_screenshot import TabScreenshotter
from ultralytics import YOLO


class MarketWorker:
    def __init__(self):
        self.tab_screenshotter = TabScreenshotter()
        self.model = YOLO("/Users/koshabbas/Desktop/work/stock_market/detect/train_run165/weights/best.pt")

    def run(self, duration_minutes=20, fps=1):
        print("Started screenshotting and detection loop.")

        # Get frame size from an initial screenshot
        initial_img = self.tab_screenshotter.capture_screenshot("3020")
        if initial_img is None:
            print("Failed to capture initial screenshot.")
            return

        width, height = initial_img.size
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_3020 = cv2.VideoWriter("3020_output.mp4", fourcc, fps, frame_size)
        out_1510 = cv2.VideoWriter("1510_output.mp4", fourcc, fps, frame_size)

        total_frames = duration_minutes * 60 * fps
        frame_count = 0

        try:
            while frame_count < total_frames:
                start_time = time.time()

                for tab, writer in [("3020", out_3020), ("1510", out_1510)]:
                    img = self.tab_screenshotter.capture_screenshot(tab)
                    if img is None:
                        print(f"Could not capture screenshot for tab {tab}")
                        continue

                    img_np = np.array(img.convert("RGB"))
                    results = self.model.predict(source=img_np, conf=0.1, iou=0.1, imgsz=1024)
                    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    boxes, scores = [], []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            if (x2 - x1) < 5 or (y2 - y1) < 15: continue
                            conf = box.conf[0].item()
                            boxes.append([x1, y1, x2, y2])
                            scores.append(conf)

                    keep = self.non_max_suppression_fast(boxes, scores, iou_thresh=0.3)
                    filtered_boxes = [boxes[i] for i in keep]
                    merged_boxes = self.merge_vertically_close_boxes(filtered_boxes)

                    for x1, y1, x2, y2 in merged_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "candle", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    writer.write(frame)
                    print(f"[{tab}] Frame {frame_count+1} | Raw: {len(boxes)}, Final: {len(merged_boxes)}")

                frame_count += 1
                time.sleep(max(0, (1.0 / fps) - (time.time() - start_time)))

        except KeyboardInterrupt:
            print("Interrupted by user.")

        finally:
            out_3020.release()
            out_1510.release()
            print("Video writers released.")

    def non_max_suppression_fast(self, boxes, scores, iou_thresh=0.4):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)  # shape (1,4)

        scores = np.array(scores)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

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
                # Check for vertical overlap and aligned x
                if abs(x1a - x1b) < x_thresh and abs(x2a - x2b) < x_thresh:
                    if abs(y1a - y2b) < y_thresh or abs(y2a - y1b) < y_thresh:
                        group.append(box2)
                        used.add(j)
            # Merge group
            xs = [b[0] for b in group] + [b[2] for b in group]
            ys = [b[1] for b in group] + [b[3] for b in group]
            merged.append([min(xs), min(ys), max(xs), max(ys)])
            used.add(i)

        return merged



if __name__ == "__main__":
    mw = MarketWorker()
    mw.run()


#Use for detecting each frame
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

