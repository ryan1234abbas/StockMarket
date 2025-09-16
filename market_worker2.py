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

    def __init__(self, model, model2, offset_x, offset_y, width, height, total_frames):
        super().__init__()
        self.model = model
        self.model2 = model2
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
        self.curr_1510 = None
        self.curr_box_1510 = None
        self.last_triggered_box = None

        #change based on speed of market
        #change based on desired buy/sell frequency
        self.buy_cooldown = 6   
        self.sell_cooldown = 6 

    def analyze_candles_tm(self, left_img, boxes_3020, labels_3020, scores_3020,
                            right_img, boxes_1510, labels_1510, scores_1510,
                            mode, candle_boxes=None, candle_labels=None, threshold=0.93, right_sz=640):
        if candle_boxes is None:
            candle_boxes = []
        if candle_labels is None:
            candle_labels = []

        # --- Initialize sticky variables if needed ---
        if not hasattr(self, 'curr_1510'):
            self.curr_1510 = None
        if not hasattr(self, 'curr_box_1510'):
            self.curr_box_1510 = None
        if not hasattr(self, 'pending_trade_executed'):
            self.pending_trade_executed = False
        if not hasattr(self, 'last_3020_pattern'):
            self.last_3020_pattern = None

        # --- Get rightmost labels and boxes ---
        rightmost_lbl_3020, box_3020, score_3020 = self.get_rightmost_label(
            boxes_3020, labels_3020, scores_3020, min_conf=0.30)
        rightmost_lbl_1510, box_1510, score_1510 = self.get_rightmost_label(
            boxes_1510, labels_1510, scores_1510, min_conf=0.30)
        current_signal = (rightmost_lbl_3020, rightmost_lbl_1510)

        # --- Get all HL/LH boxes from current detection ---
        hl_boxes = [b for b, l in zip(boxes_1510, labels_1510) if l == "HL"]
        lh_boxes = [b for b, l in zip(boxes_1510, labels_1510) if l == "LH"]
        rightmost_hl = max(hl_boxes, key=lambda b: b[0]) if hl_boxes else None
        rightmost_lh = max(lh_boxes, key=lambda b: b[0]) if lh_boxes else None

        # --- Reset sticky if 3020 pattern changes ---
        if self.last_3020_pattern != rightmost_lbl_3020:
            self.curr_1510 = rightmost_lbl_1510
            self.curr_box_1510 = box_1510
            self.pending_trade_executed = False
        self.last_3020_pattern = rightmost_lbl_3020

        # --- Sticky HL/LH logic based on 3020 ---
        if rightmost_lbl_3020 == "HH" and hl_boxes:
            # Always update both label AND box when HH pattern detected
            self.curr_1510 = "HL"
            self.curr_box_1510 = rightmost_hl
            self.pending_trade_executed = False
            print(f"Sticky set to HL with box: {self.curr_box_1510}")

        elif rightmost_lbl_3020 == "LL" and lh_boxes:
            # Always update both label AND box when LL pattern detected
            self.curr_1510 = "LH"
            self.curr_box_1510 = rightmost_lh
            self.pending_trade_executed = False
            print(f"Sticky set to LH with box: {self.curr_box_1510}")

        elif rightmost_lbl_3020 not in ("HH", "LL"):
            # Reset to current detection for non-HH/LL patterns
            self.curr_1510 = rightmost_lbl_1510
            self.curr_box_1510 = box_1510

        print(f"curr_1510: {self.curr_1510} | curr_box_1510: {self.curr_box_1510}")

        # --- Run candle detection only if valid combo ---
        if (rightmost_lbl_3020 == "HH" and self.curr_1510 == "HL") or \
        (rightmost_lbl_3020 == "LL" and self.curr_1510 == "LH"):
            candle_formation = self.model2(
                source=right_img, verbose=False, stream=False, conf=0.25, iou=0.3, imgsz=640
            )
            candle_boxes, candle_scores, candle_labels, _ = self.process_results(candle_formation)
        else:
            candle_boxes, candle_scores, candle_labels = [], [], []

        # --- Update sticky box from actual candles if available ---
        if self.curr_1510 in ("HL", "LH"):
            matching_candles = [cbox for cbox, clbl in zip(candle_boxes, candle_labels) if clbl == self.curr_1510]
            if matching_candles:
                # Update with the rightmost matching candle
                new_box = max(matching_candles, key=lambda b: b[0])
                if new_box != self.curr_box_1510:
                    self.curr_box_1510 = new_box
                    print(f"Updated sticky {self.curr_1510} box to: {self.curr_box_1510}")

        # --- Debug images ---
        debug_3020 = left_img.copy()
        debug_1510 = right_img.copy()

        # Draw 3020 detection
        if box_3020:
            x0, y0, x1, y1 = box_3020
            cv2.rectangle(debug_3020, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(debug_3020, f"{rightmost_lbl_3020} ({score_3020:.2f})", 
                    (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw 1510 detection
        if box_1510:
            x0, y0, x1, y1 = box_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(debug_1510, f"{rightmost_lbl_1510} ({score_1510:.2f})", 
                    (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw sticky box
        if self.curr_box_1510:
            x0, y0, x1, y1 = self.curr_box_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (255, 0, 0), 3)
            cv2.putText(debug_1510, f"STICKY: {self.curr_1510}", 
                    (x0, y0-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw candle detections
        for cbox, clbl in zip(candle_boxes, candle_labels):
            cx0, cy0, cx1, cy1 = cbox
            cv2.rectangle(debug_1510, (cx0, cy0), (cx1, cy1), (0, 0, 255), 2)
            cv2.putText(debug_1510, clbl, (cx0, cy0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        os.makedirs("dummy", exist_ok=True)
        cv2.imwrite("dummy/debug_3020.png", debug_3020)
        cv2.imwrite("dummy/debug_1510.png", debug_1510)

        # --- Helper: Check for black candle ---
        def has_black_candle(c_boxes, c_labels, target_box, label_type):
            if not c_boxes or not target_box:
                return False
            
            # Find the rightmost candle
            rightmost_candle = max(c_boxes, key=lambda b: (b[0] + b[2]) / 2)
            cx0, cy0, cx1, cy1 = rightmost_candle
            lx0, ly0, lx1, ly1 = target_box
            
            print(f"Rightmost candle at: {cx0}, {cx1}")

            # Check if candle is near the target box
            candle_center_x = (cx0 + cx1) // 2
            if lx0-5 <= candle_center_x <= lx1+5:
                if label_type == "HL" and cy1 < ly0:  # Candle below HL box
                    return True
                elif label_type == "LH" and cy0 > ly1:  # Candle above LH box
                    return True
            return False

        conf_3020 = f"{int(round(score_3020 * 100))}%" if score_3020 else "N/A"
        conf_1510 = f"{int(round(score_1510 * 100))}%" if score_1510 else "N/A"

        print(f"3020 Label: {rightmost_lbl_3020 or 'None'} with confidence {conf_3020}")
        print(f"1510 Label: {rightmost_lbl_1510 or 'None'} at {box_1510} with confidence {conf_1510}")

        # --- Determine BUY/SELL triggers ---
        trigger_buy = trigger_sell = False
        now = time.time()

        if rightmost_lbl_3020 == "HH" and self.curr_1510 == "HL":
            if self.curr_box_1510 and has_black_candle(candle_boxes, candle_labels, self.curr_box_1510, "HL"):
                trigger_buy = True
                print("BUY trigger: HH + HL + black candle")

        elif rightmost_lbl_3020 == "LL" and self.curr_1510 == "LH":
            if self.curr_box_1510 and has_black_candle(candle_boxes, candle_labels, self.curr_box_1510, "LH"):
                trigger_sell = True
                print("SELL trigger: LL + LH + black candle")

        def is_same_trade(box1, box2, tol=30):
            """Check if two boxes are horizontally close enough to be considered the same trade."""
            if not box1 or not box2:
                return False
            return abs(box1[0] - box2[0]) <= tol  # horizontal tolerance only

        # --- Execute BUY ---
        if mode in ("buy", "both") and trigger_buy and not self.pending_trade_executed:
            if not is_same_trade(self.last_triggered_box, self.curr_box_1510) and \
            time.time() - getattr(self, 'last_buy_time', 0) >= self.buy_cooldown:
                
                self.last_buy_time = time.time()
                self.buy_count += 1
                self.prev_trade_signal = current_signal
                self.last_triggered_box = self.curr_box_1510
                
                try:
                    buy_btn = self.cached_buy_btn or pyautogui.locateCenterOnScreen('buy_sell/buy2.png', confidence=0.8)
                    if buy_btn:
                        self.cached_buy_btn = buy_btn
                        pyautogui.click(buy_btn)
                        print("BUY executed")
                except pyautogui.ImageNotFoundException:
                    print("Buy button not found")
                
                # Reset sticky to CURRENT detection
                self.curr_1510 = rightmost_lbl_1510
                self.curr_box_1510 = box_1510
                self.pending_trade_executed = True
                return "BUY"
            else:
                print("Buy cooldown or duplicate signal, skipping")

        # --- Execute SELL ---
        elif mode in ("sell", "both") and trigger_sell and not self.pending_trade_executed:
            if not is_same_trade(self.last_triggered_box, self.curr_box_1510) and \
            time.time() - getattr(self, 'last_sell_time', 0) >= self.sell_cooldown:
                
                self.last_sell_time = time.time()
                self.sell_count += 1
                self.prev_trade_signal = current_signal
                self.last_triggered_box = self.curr_box_1510
                
                try:
                    sell_btn = self.cached_sell_btn or pyautogui.locateCenterOnScreen('buy_sell/sell2.png', confidence=0.8)
                    if sell_btn:
                        self.cached_sell_btn = sell_btn
                        pyautogui.click(sell_btn)
                        print("SELL executed")
                except pyautogui.ImageNotFoundException:
                    print("Sell button not found")
                
                # Reset sticky to CURRENT detection
                self.curr_1510 = rightmost_lbl_1510
                self.curr_box_1510 = box_1510
                self.pending_trade_executed = True
                return "SELL"
            else:
                print("Sell cooldown or duplicate signal, skipping")

        else:
            print("No valid trade signal")

        return None






    def get_rightmost_label(self,boxes, labels, scores, min_conf=0.15):
        if not boxes:
            return None, None, None

        # Sort by x0 (left to right)
        sorted_data = sorted(zip(boxes, labels, scores), key=lambda x: x[0][0])

        # Filter by confidence threshold
        filtered = [(box, label, score) for box, label, score in sorted_data if score >= min_conf]

        if not filtered:
            return None, None, None

        # Rightmost = largest x1
        rightmost = max(filtered, key=lambda x: x[0][2])
        return rightmost[1], rightmost[0], rightmost[2]


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
                        bounds = get_window_bounds("NinjaTrader 8")
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

                    '''change this based on where trading app is:
                    monitor 1 = index 0
                    monitor 2 = index 1
                    monitor 3 = index 2

                    (right now, monitor 2 is being used)
                    '''
                    full = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
                    h,w, _ = full.shape

                    left_monitor = {
                        "top": 0,
                        "left": 0,
                        "width": w//2,
                        "height": h
                    }

                    right_monitor = {
                        "top": 0,
                        "left": 0,
                        "width": w//2,
                        "height": h
                    }
                    
                    if platform.system() == "Windows":
                        trim_right = 255            # right monitor
                        trim_bottom = 80            # both monitors
                        trim_right_left_img = 150   # left monitor
                        trim_top = 30               # both monitors
                        shift_right = 40            # right monitor
                    
                    elif platform.system() == "Darwin":
                        trim_right = 300
                        trim_bottom = 220
                        trim_right_left_img = 230
                        trim_top = 50
                        shift_right = 230

                    left_img = full[
                        trim_top : h - trim_bottom,
                        : (w // 2) - trim_right_left_img,
                        :
                    ]

                    right_img = full[
                        trim_top : h - trim_bottom,
                        (w // 2 - shift_right) : (w - trim_right - shift_right),
                        :
                    ]

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
                    left_boxes, left_scores, left_labels, left_conf = self.process_results(left_results)
                    right_boxes, right_scores, right_labels, right_conf = self.process_results(right_results)

                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])
                    merged_left_labels = [left_labels[i] for i in keep_left]

                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])
                    merged_right_labels = [right_labels[i] for i in keep_right]

                    decision = self.analyze_candles_tm(
                        left_img, merged_left, merged_left_labels, left_conf,
                        right_img, merged_right, merged_right_labels, right_conf,
                        mode
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
                    conf = box.conf[0].item()
                    scores.append(conf)
                    # Convert class index to label string
                    labels.append(result.names[int(cls)])

        return boxes, scores, labels, scores

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
        #Ryan's Laptop
        # self.model = YOLO("/Users/ryanabbas/Desktop/work/StockMarket/runs/detect2/train8/weights/best.pt")
        # self.model2 = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/detect/train_19/weights/last.pt')

        #AP's Laptop
        # self.model = YOLO('/Users/Owner/StockMarket/runs/detect2/train8/weights/best.pt')
        # self.model2 = YOLO('/Users/Owner/StockMarket/runs/detect/train_19/weights/last.pt')

        # AP's main machine
        self.model = YOLO("C:/Users/ArshadParveez/Documents/Trading Code/StockMarket/runs/detect2/train8/weights/best.pt")
        self.model2 = YOLO("c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/runs/detect/train_19/weights/last.pt")

        self.app = QApplication.instance() or QApplication(sys.argv)
        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410
        self.total_frames = 20 * 60 * 1  
        
        self.detection_thread = DetectionWorker(
            model=self.model,
            model2=self.model2,
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
    mode = input("Enter mode (buy / sell / both): ").strip().lower()
    while mode not in ("buy", "sell", "both"):
        mode = input("Invalid input, enter buy, sell, or both: ").strip().lower()

    mw = MarketWorker()
    sys.exit(mw.app.exec_())