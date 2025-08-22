import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from replica_screen import ReplicaScreen
import mss
import cv2
import os
import pyautogui
import glob
import platform
import threading
import psutil

if platform.system() == "Windows":
    import msvcrt
else:
    msvcrt = None  

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
        self.prev_hhll_label   = None   # last HH or LL actually traded
        self.prev_hllh_label = None
        self.prev_hhll_box_id  = None   # (x0,y0,x1,y1) of that candle
        self.prev_hl_lh_label = None
        self.prev_box_dims = None
        self.prev_trade_signal = None
        self.counter = 0 #can't sell before buying
        self.buy_count = 0 
        self.sell_count = 0
        self.last_trade_signature = None  #avoids duplicate buying/selling in from boxes
        self.prev_label_sig = None
        self.prev_lbl_3020 = None
        self.prev_lbl_1510 = None
        self.first_trade_done = False
        #balance speed with accuracy
        self.last_trade_time = 0
        self.cached_buy_btn = None
        self.cached_sell_btn = None


        if platform.system() == "Darwin":
            self.trade_cooldown = 6.5 
        elif platform.system() == "Windows":
            self.trade_cooldown = 7
        else:
            self.trade_cooldonwn = 6.5 

        #for debugging on gui screens
        # self.update_left.emit(debug_left, [])
        # self.update_right.emit(debug_right, [])
        
        self.templates = {}

        if platform.system() == "Darwin":
            for lbl in ("HH", "LL", "HL", "LH"):
                template_files = glob.glob(f"templates/{lbl}/*.png")
                if not template_files:
                    raise FileNotFoundError(f"No template images found in templates/{lbl}/")
                self.templates[lbl] = [cv2.imread(t, cv2.IMREAD_GRAYSCALE) for t in template_files]
        
        elif platform.system() == "Windows": 
            os.makedirs("templates_windows", exist_ok=True)
            for lbl in ("HH", "LL", "HL", "LH"):
                template_files = glob.glob(f"templates_windows/{lbl}/*.png")
                if not template_files:
                    raise FileNotFoundError(f"No template images found in templates_windows/{lbl}/")
                self.templates[lbl] = [cv2.imread(t, cv2.IMREAD_GRAYSCALE) for t in template_files]
               
    def _scan_side(
        self,
        img: np.ndarray,
        merged_boxes: list,
        templates: dict[str, np.ndarray],
        want_labels: tuple[str, str],       
        debug_img: np.ndarray,
        w_crop: int = 130,
        h_crop: int = 100,
        threshold: float = 0.8,
    ):
        img_h, img_w = img.shape[:2]
        prev_y = None
        found_lbl = None

        for idx, (x0, y0, x1, y1) in enumerate(sorted(merged_boxes, key=lambda b: b[0])):
            # decide crop rectangle
            xc = (x0 + x1) // 2
            x_left = max(0, min(img_w - w_crop, xc - w_crop // 2))
            patches, boxes = [], []

            if idx == 0:
                y_above = max(0, y0 - h_crop)
                y_below = min(img_h - h_crop, y1)
                patches += [
                    ('above', img[y_above:y_above + h_crop, x_left:x_left + w_crop]),
                    ('below', img[y_below:y_below + h_crop, x_left:x_left + w_crop]),
                ]
                boxes  += [
                    (x_left, y_above, x_left + w_crop, y_above + h_crop),
                    (x_left, y_below, x_left + w_crop, y_below + h_crop),
                ]
                prev_y = y0
            else:                                
                if y0 < prev_y:
                    y_above = max(0, y0 - h_crop)
                    patches.append(('above', img[y_above:y_above + h_crop, x_left:x_left + w_crop]))
                    boxes.append((x_left, y_above, x_left + w_crop, y_above + h_crop))
                else:
                    y_below = min(img_h - h_crop, y1)
                    patches.append(('below', img[y_below:y_below + h_crop, x_left:x_left + w_crop]))
                    boxes.append((x_left, y_below, x_left + w_crop, y_below + h_crop))
                prev_y = y0

            # draw green (above) / red (below) rectangles for debug
            for (x1_, y1_, x2_, y2_) in boxes:
                color = (0, 255, 0) if y1_ < y2_ and 'above' in [p[0] for p in patches] else (0, 0, 255)
                cv2.rectangle(debug_img, (x1_, y1_), (x2_, y2_), color, 2)

            # template matching
            for pos, patch in patches:
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                for lbl in want_labels:          
                    max_val_for_label = 0
                    # self.templates[lbl] is a list of templates for that label
                    for lbl in want_labels:
                        max_val_for_label = 0
                        for tmpl in self.templates[lbl]:
                            res = cv2.matchTemplate(patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                            max_val = res.max()
                            if max_val > max_val_for_label:
                                max_val_for_label = max_val

                        if max_val_for_label >= threshold:
                            label_main = lbl
                            break
        
                if found_lbl:
                    break
            if found_lbl:
                break

        return found_lbl

    def scan_rightmost_candle(self, img, boxes, want_labels, debug_img, label_side, threshold=0.93):
        img_h, img_w = img.shape[:2]

        if not boxes:
            print(f"{label_side}: No candles detected.")
            return (None, None), ([], None), debug_img  # empty list instead of single label

        # Pick rightmost box but stabilize if previous exists
        # Always default to the actual rightmost box
        rightmost_box = max(boxes, key=lambda b: b[0])

        # Try to stabilize with previous box, if available
        if hasattr(self, 'prev_candle_box') and self.prev_candle_box is not None:
            prev_x0 = self.prev_candle_box[0]
            candidate_box = min(boxes, key=lambda b: abs(b[0] - prev_x0))

            # Only override if the candidate is reasonably close to previous
            if abs(candidate_box[0] - prev_x0) < 50:
                rightmost_box = candidate_box

        x0, y0, x1, y1 = rightmost_box

        scan_margin = 35
        right_edge_buffer = 70
        scan_x0 = max(0, x1 - scan_margin)
        scan_x1 = img_w - right_edge_buffer

        center_y = (y0 + y1) // 2
        box_height = 5000  # or any default

        scan_y0 = max(0, center_y - box_height // 2)
        scan_y1 = min(img_h, center_y + box_height // 2)

        # Now apply extra height expansion
        extra_height = 1000
        scan_y0 = max(0, scan_y0 - extra_height // 2)
        scan_y1 = min(img_h, scan_y1 + extra_height // 2)

        if scan_y1 - scan_y0 < 100:
            scan_y0 = max(0, scan_y1 - 100)

        if platform.system() == "Darwin":  # macOS
            scan_x0 = max(0, x1 - scan_margin)  
            scan_x1 = img_w 
            
        elif platform.system() == "Windows":
            scan_x1 = img_w
            scan_x0 += 20
            threshold = 0.93

        cv2.rectangle(debug_img, (scan_x0, scan_y0), (scan_x1, scan_y1), (255, 0, 0), 2)

        patch = img[scan_y0:scan_y1, scan_x0:scan_x1]
        matches = []  # list to hold (label, box) tuples

        if patch.size > 0:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            os.makedirs("dummy", exist_ok=True)
            cv2.imwrite(f"dummy/bluebox_{label_side}.png", patch)

            for label in want_labels:
                max_conf = 0
                best_loc = None
                for tmpl in self.templates[label]:
                    # Skip template if bigger than patch
                    if gray_patch.shape[0] < tmpl.shape[0] or gray_patch.shape[1] < tmpl.shape[1]:
                        continue
                    res = cv2.matchTemplate(gray_patch, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, curr_conf, _, max_loc = cv2.minMaxLoc(res)
                    if curr_conf > max_conf:
                        max_conf = curr_conf
                        best_loc = max_loc  # location of best match for this template

                if max_conf >= threshold and best_loc is not None:
                    # Calculate absolute box coordinates of detected label inside img
                    tmpl_h, tmpl_w = self.templates[label][0].shape[:2]
                    abs_x0 = scan_x0 + best_loc[0]
                    abs_y0 = scan_y0 + best_loc[1]
                    abs_x1 = abs_x0 + tmpl_w
                    abs_y1 = abs_y0 + tmpl_h

                    matches.append((label, (abs_x0, abs_y0, abs_x1, abs_y1)))

                    #print(f"{label_side}: matched {label} with confidence {max_conf:.2f}")

        self.prev_candle_box = rightmost_box

        # Return list of all matched labels with their boxes, plus the rightmost candle box for reference
        return (None, None), (matches, rightmost_box), debug_img


    def analyze_candles_tm(self, left_img, merged_left, right_img, merged_right, templates, threshold=0.93, w_crop=130, h_crop=100):

        def box_width(box):
            return (box[2] - box[0]) if box else 0

        def get_rightmost_label(labels):
            if not labels:
                return None
            rightmost_x1 = max(lb[1][2] for lb in labels)
            candidates = [lb for lb in labels if lb[1][2] == rightmost_x1]
            return candidates[0][0]

        def is_label_latest_by_coords(labels_with_boxes, desired_label):
            return get_rightmost_label(labels_with_boxes) == desired_label

        now = time.time()
        if now - getattr(self, 'last_trade_time', 0) < getattr(self, 'trade_cooldown', 0):
            print("Cooldown Active.")
            return None

        left_result = {'labels': [], 'box': None, 'debug': None}
        right_result = {'labels': [], 'box': None, 'debug': None}

        def scan_left():
            (_, _), (labels, box), debug = self.scan_rightmost_candle(
                left_img, merged_left, ("HH", "LL", "HL", "LH"), left_img.copy(), "3020", threshold)
            left_result.update({'labels': labels or [], 'box': box, 'debug': debug})

        def scan_right():
            (_, _), (labels, box), debug = self.scan_rightmost_candle(
                right_img, merged_right, ("HH", "LL", "HL", "LH"), right_img.copy(), "1510", threshold)
            right_result.update({'labels': labels or [], 'box': box, 'debug': debug})

        t_left = threading.Thread(target=scan_left)
        t_right = threading.Thread(target=scan_right)
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

        labels_3020, box_3020, debug_3020 = left_result['labels'], left_result['box'], left_result['debug']
        labels_1510, box_1510, debug_1510 = right_result['labels'], right_result['box'], right_result['debug']

        #   Width tracking  
        curr_width_3020 = box_width(box_3020)
        curr_width_1510 = box_width(box_1510)

        prev_width_3020 = getattr(self, 'prev_width_3020', None)
        prev_width_1510 = getattr(self, 'prev_width_1510', None)

        new_3020_candle = (prev_width_3020 is None) or (curr_width_3020 < prev_width_3020)
        new_1510_candle = (prev_width_1510 is None) or (curr_width_1510 < prev_width_1510)

        #   Save debug images  
        trim_amount = 100
        debug_3020 = debug_3020[trim_amount:, :] if debug_3020 is not None else None
        debug_1510 = debug_1510[trim_amount:, :] if debug_1510 is not None else None
        os.makedirs("dummy", exist_ok=True)
        if debug_3020 is not None:
            cv2.imwrite("dummy/debug_3020.png", debug_3020)
        if debug_1510 is not None:
            cv2.imwrite("dummy/debug_1510.png", debug_1510)

        #   Update widths  
        self.prev_width_3020 = curr_width_3020
        self.prev_width_1510 = curr_width_1510

        #   Proceed only if both boxes exist  
        if not box_3020 or not box_1510:
            return None

        #   Reset stored labels if new candle box detected  
        curr_box_dims = (box_3020[0], box_3020[1], box_3020[2], box_3020[3])
        if not hasattr(self, 'prev_box_dims') or self.prev_box_dims is None:
            self.prev_box_dims = curr_box_dims
            self.prev_lbl_3020 = None
            self.prev_lbl_1510 = None
            self.prev_trade_signal = None
        else:
            if curr_width_3020 < (self.prev_box_dims[2] - self.prev_box_dims[0]):
                print("New candle box detected, resetting last labels.")
                self.prev_lbl_3020 = None
                self.prev_lbl_1510 = None
                self.prev_trade_signal = None
            self.prev_box_dims = curr_box_dims

        #   Determine current rightmost labels for debug  
        rightmost_lbl_3020 = get_rightmost_label(labels_3020)
        rightmost_lbl_1510 = get_rightmost_label(labels_1510)
        current_signal = (rightmost_lbl_3020, rightmost_lbl_1510)

        #debug statements
        print(f"3020 Label: {rightmost_lbl_3020 or 'None'}")
        print(f"1510 Label: {rightmost_lbl_1510 or 'None'}")

        #Trading logic  
        if (is_label_latest_by_coords(labels_3020, "HH") and
            is_label_latest_by_coords(labels_1510, "HL") and
            (new_3020_candle or self.prev_lbl_3020 != "HH") and
            (new_1510_candle or self.prev_lbl_1510 != "HL")):

            if (current_signal != getattr(self, 'prev_trade_signal', None)):
                self.last_trade_time = now
                print(">> BUY signal detected (HH on 3020, HL on 1510)")
                self.buy_count = getattr(self, 'buy_count', 0) + 1
                self.counter = getattr(self, 'counter', 0) + 1
                self.prev_lbl_3020 = "HH"
                self.prev_lbl_1510 = "HL"
                self.prev_trade_signal = current_signal
                
                buy_btn = self.cached_buy_btn
                if buy_btn is None:  
                    buy_btn = pyautogui.locateCenterOnScreen('buy_sell/buy.png', confidence=0.8)
                    if buy_btn:
                        self.cached_buy_btn = buy_btn

                if buy_btn:
                    pyautogui.click(buy_btn)
                else:
                    print("Buy Transaction Failed")
                return "BUY"
            else:
                print("Duplicate BUY signal, ignoring.")

        elif (is_label_latest_by_coords(labels_3020, "LL") and 
            is_label_latest_by_coords(labels_1510, "LH") and
            (new_3020_candle or self.prev_lbl_3020 != "LL") and
            (new_1510_candle or self.prev_lbl_1510 != "LH")):

            if getattr(self, 'counter', 0) > 0:
                if (current_signal != getattr(self, 'prev_trade_signal', None)):
                    self.last_trade_time = now
                    print(">> SELL signal detected (LL on 3020, LH on 1510)")
                    self.sell_count = getattr(self, 'sell_count', 0) + 1
                    self.counter -= 1
                    self.prev_lbl_3020 = "LL"
                    self.prev_lbl_1510 = "LH"
                    self.prev_trade_signal = current_signal
                    
                    sell_btn = self.cached_sell_btn
                    if sell_btn is None:  
                        sell_btn = pyautogui.locateCenterOnScreen('buy_sell/sell.png', confidence=0.8)
                        if sell_btn:
                            self.cached_sell_btn = sell_btn

                    if sell_btn:
                        pyautogui.click(sell_btn)
                    else:
                        print("Sell Transaction Failed")
                    return "SELL"
                else:
                    print("Duplicate SELL signal, ignoring.")
            else:
                print("Cannot SELL before BUY.")

        print("No new candle detected or no valid trade signal.")
        return None

    def preprocess_for_ocr(patch):
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # enhance text visibility
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    
    def classify_patch(self, main_patch_gray):
        best_label = None
        best_confidence = 0
        threshold = 0.93  

        for lbl in self.templates:
            for tmpl in self.templates[lbl]:
                res = cv2.matchTemplate(main_patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > best_confidence and max_val >= threshold:
                    best_label = lbl
                    best_confidence = max_val

        if best_label:
            print(f"Matched {best_label} with confidence {best_confidence:.2f}")

        return best_label, best_confidence

    def crop(self,
            img: np.ndarray,
            box, # [x1, y1, x2, y2]
            width: int | None = None,
            height: int = 200,
            pad_x: int = 10,
            side: str = "above") -> np.ndarray:

        h, w = img.shape[:2]
        x1, y1, x2, y2 = box

        # horizontal limits 
        if width is None:
            width = (x2 - x1) + 2 * pad_x
        left = max(0, x1 - pad_x)
        right = min(w, left + width)
        left = max(0, right - width) # recompute if we clipped on the right

        # vertical limits 
        if side == "above":
            top = max(0, y1 - height)
            bottom = y1
        else:  # below
            top = y2
            bottom = min(h, y2 + height)
            top = bottom - height        

        return img[top:bottom, left:right].copy()


    def run(self):

        #   Key press detection  
        if os.name == "posix":
            import sys, select, tty, termios

            # Save original terminal settings
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # non-canonical mode

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
                    res = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                    # Parse res.stdout to x, y, width, height
                    # Placeholder fallback for now:
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
                        else:
                            self.offset_x, self.offset_y, self.width, self.height = get_window_bounds("Media Player")
                            
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
                    left_boxes, left_scores = self.process_results(left_results)
                    right_boxes, right_scores = self.process_results(right_results)

                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])

                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])

                    decision = self.analyze_candles_tm(left_img, merged_left, right_img, merged_right, self.templates)

                    if decision:
                        print(f"Trade decision: {decision}")
                    print(f"Number of buys: {self.buy_count}")
                    print(f"Number of sells: {self.sell_count}")

                    # Draw & emit
                    left_img = self.draw_coords_only(left_img, merged_left)
                    right_img = self.draw_coords_only(right_img, merged_right)
                    self.update_left.emit(left_img, merged_left)
                    self.update_right.emit(right_img, merged_right)

                    # Frame stats
                    self.frame_count += 1
                    frame_processing_time = time.time() - start_time
                    total_processing_time += frame_processing_time
                    avg_processing_time = total_processing_time / self.frame_count

                    print(f"\nFrame {self.frame_count} processed in {frame_processing_time:.2f} sec.")
                    
                    if self.frame_count % 10 == 0:
                        usage = psutil.cpu_percent(interval=1)
                        freq = psutil.cpu_freq()
                        print(f"CPU: {usage}% | {freq.current:.0f} MHz / {freq.max:.0f} MHz")

                    time.sleep(0.0001)

                    #stop program
                    key = get_key()  # works for both Windows and macOS
                    if key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        minutes, seconds = divmod(total_processing_time, 60)
                        print(f"Runtime: {int(minutes)} min {seconds:.2f} sec")
                        print(f"Average runtime per frame: {avg_processing_time:.2f} seconds")
                        print(f"Final number of buys: {self.buy_count}")
                        print(f"Final number of sells: {self.sell_count}")
                        break

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting...")
            finally:
                self.finished.emit()


    def process_results(self, results):
        boxes = []
        scores = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Skip small boxes
                if (x2 - x1) < 4 or (y2 - y1) < 20:
                    continue

                # Clamp coordinates to be within image bounds
                h, w = result.orig_shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                # Only keep valid boxes
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(box.conf[0].item())
        return boxes, scores

    
    '''Uncomment lines to observe coords at each detected candle'''
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
        #Ryan's IMAC
        #self.model = YOLO('/Users/koshabbas/Desktop/work/stock_market/runs/detect/train_19/weights/last.pt')
        
        #Ryan's Laptop
        #self.model = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/detect/train_19/weights/last.pt')
        
        #AP's Laptop
        self.model = YOLO('/Users/Owner/StockMarket/runs/detect/train_19/weights/last.pt')
        
        self.app = QApplication.instance() or QApplication(sys.argv)

        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410

        self.total_frames = 20 * 60 * 1  
        
        '''
        Uncomment lines below to observe candle detection
        '''
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