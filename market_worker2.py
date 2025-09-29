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

        import time, os, cv2, pyautogui

        if candle_boxes is None:
            candle_boxes = []
        if candle_labels is None:
            candle_labels = []

        # --- Initialize sticky vars ---
        if not hasattr(self, 'curr_1510'):
            self.curr_1510 = None
        if not hasattr(self, 'curr_box_1510'):
            self.curr_box_1510 = None
        if not hasattr(self, 'pending_trade_executed'):
            self.pending_trade_executed = False
        if not hasattr(self, 'last_3020_pattern'):
            self.last_3020_pattern = None

        # --- Get rightmost labels ---
        rightmost_lbl_3020, box_3020, score_3020 = self.get_rightmost_label(
            boxes_3020, labels_3020, scores_3020, min_conf=0.30)
        rightmost_lbl_1510, box_1510, score_1510 = self.get_rightmost_label(
            boxes_1510, labels_1510, scores_1510, min_conf=0.30)
        current_signal = (rightmost_lbl_3020, rightmost_lbl_1510)

        # --- HL/LH boxes ---
        hl_boxes = [b for b, l in zip(boxes_1510, labels_1510) if l == "HL"]
        lh_boxes = [b for b, l in zip(boxes_1510, labels_1510) if l == "LH"]
        rightmost_hl = max(hl_boxes, key=lambda b: b[0]) if hl_boxes else None
        rightmost_lh = max(lh_boxes, key=lambda b: b[0]) if lh_boxes else None

        # --- Sticky update logic (look-back aware) ---
        if self.curr_1510 is None:
            # First-time init
            self.curr_1510 = rightmost_lbl_1510
            self.curr_box_1510 = box_1510

        elif self.pending_trade_executed:
            # Reset after trade executed
            self.pending_trade_executed = False

        else:
            # Only update sticky if trend changes in 1510 relative to 3020
            if self.curr_1510 in ("HL", "LH"):
                matches = [cbox for cbox, clbl in zip(candle_boxes, candle_labels) if clbl == self.curr_1510]
                if matches:
                    # Keep sticky, pick rightmost matching candle
                    self.curr_box_1510 = max(matches, key=lambda b: b[0])
                else:
                    # Sticky invalid, assign new detection
                    self.curr_1510 = rightmost_lbl_1510
                    self.curr_box_1510 = box_1510
            else:
                # If current sticky is None or not HL/LH, assign current detection
                self.curr_1510 = rightmost_lbl_1510
                self.curr_box_1510 = box_1510

        conf_3020 = f"{int(round(score_3020 * 100))}%" if score_3020 else "N/A" 
        conf_1510 = f"{int(round(score_1510 * 100))}%" if score_1510 else "N/A" 
        print(f"3020 Label: {rightmost_lbl_3020 or 'None'} with confidence {conf_3020}") 
        print(f"1510 Label: {rightmost_lbl_1510 or 'None'} at {box_1510} with confidence {conf_1510}")

        # Always update last 3020
        self.last_3020_pattern = rightmost_lbl_3020

        # --- Candle detection ---
        results = self.model(source=right_img, verbose=False, stream=False, conf=0.20, iou=0.3, imgsz=640)
        candle_boxes, candle_scores, candle_labels, _ = self.process_results(results)

        candle_x0_x1 = []
        if candle_boxes:
            rightmost_candle = max(candle_boxes, key=lambda b: b[0])
            cx0, cy0, cx1, cy1 = rightmost_candle
            candle_x0_x1.extend([cx0, cx1])

        # --- Sync sticky box with candles ---
        if self.curr_1510 in ("HL", "LH"):
            matches = [cbox for cbox, clbl in zip(candle_boxes, candle_labels) if clbl == self.curr_1510]
            if matches:
                new_box = max(matches, key=lambda b: b[0])
                if new_box != self.curr_box_1510:
                    self.curr_box_1510 = new_box

        # --- Debug drawing ---
        debug_3020, debug_1510 = left_img.copy(), right_img.copy()
        if box_3020:
            x0, y0, x1, y1 = box_3020
            cv2.rectangle(debug_3020, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(debug_3020, f"{rightmost_lbl_3020} ({score_3020:.2f})", (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if rightmost_lbl_1510 and box_1510:
            x0, y0, x1, y1 = box_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (0, 255, 255), 4)
            cv2.putText(debug_1510, f"RAW: {rightmost_lbl_1510} ({score_1510:.2f})", (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if self.curr_box_1510:
            x0, y0, x1, y1 = self.curr_box_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(debug_1510, f"STICKY: {self.curr_1510}", (x0, y0-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        os.makedirs("dummy", exist_ok=True)
        cv2.imwrite("dummy/debug_3020.png", debug_3020)
        cv2.imwrite("dummy/debug_1510.png", debug_1510)

        # --- Trade helper ---
        def has_black_candle(c_x0, c_x1, c_labels, target_box, label_type):
            if not target_box:
                return False
            lx0, ly0, lx1, ly1 = target_box
            center_x = (c_x0 + c_x1) // 2
            if lx0 <= center_x <= lx1:
                if label_type == "HL" and cy1 < ly0:
                    return True
                elif label_type == "LH" and cy0 > ly1:
                    return True
            return False

        # --- Determine triggers ---
        trigger_buy = trigger_sell = False
        if self.curr_1510 == "HL" and rightmost_lbl_3020 == "HH":
            if self.curr_box_1510 and has_black_candle(candle_x0_x1[0], candle_x0_x1[1], candle_labels, self.curr_box_1510, "HL"):
                trigger_buy = True
        elif self.curr_1510 == "LH" and rightmost_lbl_3020 == "LL":
            if self.curr_box_1510 and has_black_candle(candle_x0_x1[0], candle_x0_x1[1], candle_labels, self.curr_box_1510, "LH"):
                trigger_sell = True

        # --- Execute trades ---
        def is_same_trade(box1, box2, tol=30):
            if not box1 or not box2:
                return False
            return abs(box1[0]-box2[0]) <= tol

        now = time.time()

        if mode in ("buy", "both") and trigger_buy and not self.pending_trade_executed:
            if not is_same_trade(getattr(self, 'last_triggered_box', None), self.curr_box_1510) and \
            now - getattr(self, 'last_buy_time', 0) >= self.buy_cooldown:

                self.last_buy_time = now
                self.buy_count += 1
                self.prev_trade_signal = current_signal
                self.last_triggered_box = self.curr_box_1510
                self.pending_trade_executed = True

                try:
                    buy_btn = self.cached_buy_btn or pyautogui.locateCenterOnScreen('buy_sell/buy2.png', confidence=0.8)
                    if buy_btn:
                        self.cached_buy_btn = buy_btn
                        pyautogui.click(buy_btn)
                        print("BUY executed")
                except pyautogui.ImageNotFoundException:
                    print("Buy button not found")

                return "BUY"

        elif mode in ("sell", "both") and trigger_sell and not self.pending_trade_executed:
            if not is_same_trade(getattr(self, 'last_triggered_box', None), self.curr_box_1510) and \
            now - getattr(self, 'last_sell_time', 0) >= self.sell_cooldown:

                self.last_sell_time = now
                self.sell_count += 1
                self.prev_trade_signal = current_signal
                self.last_triggered_box = self.curr_box_1510
                self.pending_trade_executed = True

                try:
                    sell_btn = self.cached_sell_btn or pyautogui.locateCenterOnScreen('buy_sell/sell2.png', confidence=0.8)
                    if sell_btn:
                        self.cached_sell_btn = sell_btn
                        pyautogui.click(sell_btn)
                        print("SELL executed")
                except pyautogui.ImageNotFoundException:
                    print("Sell button not found")

                return "SELL"

        return None

   
    def get_rightmost_label(self, boxes, labels, scores, min_conf=0.30):
        valid_labels = {"HH", "LL", "HL", "LH"}

        if not boxes or not labels or not scores:
            return None, None, None

        # Only keep boxes with valid labels + above confidence
        filtered = [
            (b, l, s) for b, l, s in zip(boxes, labels, scores)
            if l in valid_labels and s >= min_conf
        ]

        if not filtered:
            return None, None, None

        # Pick the rightmost among valid ones
        box, label, score = max(filtered, key=lambda x: x[0][0])  # x[0][0] = leftmost x coord
        return label, box, score


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
        self.model = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt')

        #AP's Laptop
        # self.model = YOLO('/Users/Owner/StockMarket/runs/detect2/train8/weights/best.pt')

        # AP's main machine
        #self.model = YOLO("C:/Users/ArshadParveez/Documents/Trading Code/StockMarket/runs/detect2/train8/weights/best.pt")

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
    mode = input("Enter mode (buy / sell / both): ").strip().lower()
    while mode not in ("buy", "sell", "both"):
        mode = input("Invalid input, enter buy, sell, or both: ").strip().lower()

    mw = MarketWorker()
    sys.exit(mw.app.exec_())