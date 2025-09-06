import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import mss
import cv2
import os
import pyautogui
import platform
from collections import deque
from datetime import date, datetime
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
        self.running = True
        self.prev_box_dims = None
        self.prev_trade_signal = None
        self.counter = 0 
        self.buy_count = 0 
        self.sell_count = 0
        self.prev_lbl_3020 = None
        self.prev_lbl_1510 = None
        self.last_trade_time = 0
        self.cached_buy_btn = None
        self.cached_sell_btn = None
        self.last_buy_time = 0
        self.last_sell_time = 0
        self.histories = {
            '3020': deque(maxlen=2),
            '1510': deque(maxlen=2)
        }
        self.prev_labels = {
            '3020': {'x0': None, 'label': None},
            '1510': {'x0': None, 'label': None}
        }

        #change based on speed of market
        #change based on desired buy/sell frequency
        self.buy_cooldown = 8.5   
        self.sell_cooldown = 8.5  

    def analyze_candles_tm(self, left_img, boxes_3020, labels_3020,
                        right_img, boxes_1510, labels_1510,
                        threshold=0.93):
        """
        Analyze candles using YOLO detections and determine BUY/SELL signals.
        """

        def get_rightmost_label(boxes, labels, side):
            if not boxes:
                return None, None

            idx = np.argmax([b[2] for b in boxes])
            curr_label, curr_box = labels[idx], boxes[idx]
            curr_x0 = curr_box[0]

            hist = self.histories[side]  # class-level deque
            hist.append(curr_x0)

            # Initialize a persistent flag dict if needed
            if not hasattr(self, 'label_blocked'):
                self.label_blocked = {'3020': False, '1510': False}

            # Check for sudden drop
            if len(hist) >= 2:
                if (hist[-1] - hist[-2]) < -100:
                    self.label_blocked[side] = True

            # If blocked, check for recovery
            if self.label_blocked[side]:
                # Only unblock if it goes back up by ≥100 px from the lowest in the blocked period
                min_blocked_x = min(list(hist)[-2:])  # last two values during block
                if curr_x0 - min_blocked_x >= 100:
                    self.label_blocked[side] = False
                else:
                    curr_label = None  # still blocked
            
            return curr_label, curr_box

        # Get rightmost label per monitor
        rightmost_lbl_3020, box_3020 = get_rightmost_label(boxes_3020, labels_3020, '3020')
        rightmost_lbl_1510, box_1510 = get_rightmost_label(boxes_1510, labels_1510, '1510')
        current_signal = (rightmost_lbl_3020, rightmost_lbl_1510)

        # Width tracking to detect new candle
        def box_width(box):
            return (box[2] - box[0]) if box else 0

        curr_width_3020 = box_width(box_3020)
        curr_width_1510 = box_width(box_1510)

        # Save debug images with rightmost boxes
        debug_3020 = left_img.copy()
        debug_1510 = right_img.copy()

        if box_3020 is not None:
            x0, y0, x1, y1 = box_3020
            cv2.rectangle(debug_3020, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(debug_3020, rightmost_lbl_3020, (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if box_1510 is not None:
            x0, y0, x1, y1 = box_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(debug_1510, rightmost_lbl_1510, (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        os.makedirs("dummy", exist_ok=True)
        cv2.imwrite("dummy/debug_3020.png", debug_3020)
        cv2.imwrite("dummy/debug_1510.png", debug_1510)

        # Update previous widths
        self.prev_width_3020 = curr_width_3020
        self.prev_width_1510 = curr_width_1510

        # Reset stored labels if new candle detected
        curr_box_dims = (box_3020[0], box_3020[1], box_3020[2], box_3020[3]) if box_3020 else None
        if not hasattr(self, 'prev_box_dims') or self.prev_box_dims is None:
            self.prev_box_dims = curr_box_dims
            self.prev_lbl_3020 = None
            self.prev_lbl_1510 = None
            self.prev_trade_signal = None
        else:
            if box_3020 and curr_width_3020 < (self.prev_box_dims[2] - self.prev_box_dims[0]):
                self.prev_lbl_3020 = None
                self.prev_lbl_1510 = None
                self.prev_trade_signal = None
            self.prev_box_dims = curr_box_dims

        # Debug prints
        print(f"3020 Label: {rightmost_lbl_3020 or 'None'}")
        print(f"1510 Label: {rightmost_lbl_1510 or 'None'}")

        now = time.time()

        # BUY logic
        if rightmost_lbl_3020 == "HH" and rightmost_lbl_1510 == "HL":
            if (current_signal != getattr(self, 'prev_trade_signal', None)
                and now - getattr(self, 'last_buy_time', 0) >= self.buy_cooldown):

                self.last_buy_time = now
                self.buy_count = getattr(self, 'buy_count', 0) + 1
                self.counter = getattr(self, 'counter', 0) + 1
                self.prev_lbl_3020 = "HH"
                self.prev_lbl_1510 = "HL"
                self.prev_trade_signal = current_signal

                try:
                    buy_btn = self.cached_buy_btn or pyautogui.locateCenterOnScreen(
                        'buy_sell/buy.png', confidence=0.8
                    )
                    if buy_btn:
                        self.cached_buy_btn = buy_btn
                        pyautogui.click(buy_btn)
                except pyautogui.ImageNotFoundException:
                    pass  
                return "BUY"

            elif now - getattr(self, 'last_buy_time', 0) < self.buy_cooldown:
                print("Buy cooldown active.")
            else:
                print("Duplicate BUY signal, ignoring.")

        # SELL logic
        elif rightmost_lbl_3020 == "LL" and rightmost_lbl_1510 == "LH":
            if (self.counter > 0
                and current_signal != getattr(self, 'prev_trade_signal', None)
                and now - getattr(self, 'last_sell_time', 0) >= self.sell_cooldown):

                self.last_sell_time = now
                self.sell_count += 1
                self.counter -= 1
                self.prev_lbl_3020 = "LL"
                self.prev_lbl_1510 = "LH"
                self.prev_trade_signal = current_signal

                try:
                    sell_btn = self.cached_sell_btn or pyautogui.locateCenterOnScreen(
                        'buy_sell/sell.png', confidence=0.8
                    )
                    if sell_btn:
                        self.cached_sell_btn = sell_btn
                        pyautogui.click(sell_btn)
                except pyautogui.ImageNotFoundException:
                    pass  
                return "SELL"

            elif self.counter == 0:
                print("Cannot SELL before BUY.")
            elif now - getattr(self, 'last_sell_time', 0) < self.sell_cooldown:
                print("Sell cooldown active.")
            else:
                print("Duplicate SELL signal, ignoring.")

        # No valid trade signal
        else:
            print("No valid trade signal.")

        return None

    def run(self):

        # Key press detection  
        if os.name == "posix":
            import sys, select, tty, termios

            # Save original terminal settings
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def key_pressed():
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                return dr != []

            def get_key():
                if key_pressed():
                    return sys.stdin.read(1).lower()
                return None

            import atexit
            # Restore terminal settings on exit
            atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings))
        else:
            import msvcrt
            def get_key():
                if msvcrt.kbhit():
                    return msvcrt.getch().decode("utf-8").lower()
                return None

        total_processing_time = 0

        def get_window_bounds(title):
            """Detect window position and size dynamically per OS"""
            system = platform.system()
            if system == "Windows":
                try:
                    import pygetwindow as gw
                    win = gw.getWindowsWithTitle(title)
                    if win:
                        w = win[0]
                        return w.left, w.top, w.width, w.height
                except Exception:
                    return 0, 0, 800, 600  #fallback default
            elif system == "Darwin":
                # macOS: use AppleScript
                import subprocess
                script = f'''
                tell application "System Events"
                    tell application process "{title}"
                        set frontmost to true
                        tell window 1
                            set {{"xPos:", position, "sizeVal:", size}}
                        end tell
                    end tell
                end tell
                '''
                try:
                    return 0, 0, 1300, 1300
                except Exception:
                    return 0, 0, 1300, 1300
            else:
                # Linux fallback
                return 0, 0, 800, 600

        with mss.mss() as sct:
            try:
                while self.running:
                    start_time = time.time()

                    #  Detect app window dynamically 
                    if platform.system() == "Darwin":
                        self.offset_x, self.offset_y, self.width, self.height = get_window_bounds("QuickTime Player")
                    else:
                        bounds = get_window_bounds("NinjaTrader")
                        if bounds:
                            self.offset_x, self.offset_y, self.width, self.height = bounds
                    #    else:
                    #        self.offset_x, self.offset_y, self.width, self.height = get_window_bounds("Media Player")
                            
                    #  Define dynamic monitor regions 
                    trim_right_ratio = 0.30   
                    trim_bottom_ratio = 0.47
                    if platform.system() == "Windows":
                        extra_height_ratio = 0.6  
                    else:
                        extra_height_ratio = 0  

                    shift_left_ratio = 0.2 

                    left_monitor = {
                        "top": self.offset_y,
                        "left": self.offset_x,
                        "width": int(self.width * 0.5 * (1 - trim_right_ratio)),
                        "height": int(self.height * (1 + extra_height_ratio) * (1 - trim_bottom_ratio))
                    }

                    right_monitor = {
                        "top": self.offset_y,
                        "left": self.offset_x + int(self.width * 0.5 * (1 - shift_left_ratio)),  # shift left
                        "width": int(self.width * 0.5 * (1 - trim_right_ratio)),  # trimmed width
                        "height": int(self.height * (1 + extra_height_ratio) * (1 - trim_bottom_ratio))
                    }

                    #  Grab screenshots 
                    left_img = np.array(sct.grab(left_monitor))[:, :, :3]
                    right_img = np.array(sct.grab(right_monitor))[:, :, :3]

                    #  Resize for model 
                    m32 = lambda v: ((v + 31) // 32) * 32
                    left_sz = (m32(left_monitor['width']), m32(left_monitor['height']))
                    right_sz = (m32(right_monitor['width']), m32(right_monitor['height']))

                    # Model predictions 
                    left_results = self.model.predict(
                        source=left_img, verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=left_sz)
                    right_results = self.model.predict(
                        source=right_img, verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=right_sz)

                    # Process results 
                    left_boxes, left_scores, left_labels = self.process_results(left_results)
                    right_boxes, right_scores, right_labels = self.process_results(right_results)

                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])
                    merged_left_labels = [left_labels[i] for i in keep_left]

                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])
                    merged_right_labels = [right_labels[i] for i in keep_right]

                    decision = self.analyze_candles_tm(
                        left_img, merged_left, merged_left_labels,
                        right_img, merged_right, merged_right_labels
                    )

                    if decision:
                        print(f"Trade decision: {decision}")
                    print(f"Number of buys: {self.buy_count}")
                    print(f"Number of sells: {self.sell_count}")


                    # Frame stats
                    self.frame_count += 1
                    frame_processing_time = time.time() - start_time
                    total_processing_time += frame_processing_time
                    avg_processing_time = total_processing_time / self.frame_count

                    print(f"\nFrame {self.frame_count} processed in {frame_processing_time:.2f} sec.")
                    time.sleep(0.0001)

                    #stop program
                    key = get_key()  # works for both Windows and macOS
                    if key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        minutes, seconds = divmod(total_processing_time, 60)

                        current_time = datetime.now().strftime('%H:%M')
                        current_date = str(date.today())

                        # Format log content
                        log_content = (
                            f"\nTime: {current_time}  Date: {current_date}\n"
                            f"Runtime: {int(minutes)} min {seconds:.2f} sec\n"
                            f"Average runtime per frame: {avg_processing_time:.2f} seconds\n"
                            f"Final number of buys: {self.buy_count}\n"
                            f"Final number of sells: {self.sell_count}\n"
                        )

                        print(log_content)

                        with open("log.txt", "a") as log_file:  # Use "w" instead of "a" to overwrite each time
                            log_file.write(log_content)
                        break

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting...")
            finally:
                self.finished.emit()

    def process_results(self, results):
        boxes = []
        scores = []
        labels = []
        
        for result in results:
            for box, cls in zip(result.boxes, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Clamp coordinates
                h, w = result.orig_shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(box.conf[0].item())
                    # Convert class index to label string
                    labels.append(result.names[int(cls)])

        return boxes, scores, labels

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
        #Ryan's IMAC
        
        #Ryan's Laptop
        #self.model = YOLO("/Users/ryanabbas/Desktop/work/StockMarket/runs/detect2/train8/weights/best.pt")
        
        #AP's Laptop
        #self.model = YOLO('/Users/Owner/StockMarket/runs/detect2/train8/weights/best.pt')
        
        #AP's main machine
        self.model = YOLO("C:/Users/home/OneDrive/Desktop/StockMarket/runs/detect2/train8/weights/best.pt")

        self.app = QApplication.instance() or QApplication(sys.argv)
        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410
        self.total_frames = 20 * 60 * 1  
        
        self.detection_thread = DetectionWorker(
            model=self.model,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames
        )
        
        self.detection_thread.finished.connect(self.on_finished)
        self.detection_thread.start()

    def on_finished(self):
        print("Detection finished.")
        self.app.quit()

if __name__ == "__main__":
    mw = MarketWorker()
    sys.exit(mw.app.exec_())