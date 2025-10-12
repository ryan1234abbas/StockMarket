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
import torch

# GPU configuration
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


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
        self.pending_srl_trade = None
        self.mode = mode
        self.prev_rml_1510 = None
        self.prev_srl_1510 = None
        self.last_correct_srl_label = None
        self.last_correct_srl_x = None
        self.last_correct_rml_x = None
        self.srl_lockout_after_trade = False
        self.last_correct_3020_rml_x = None
        self.is_first_frame = True
        self.last_rml_1510_x = None
        self.rml_backward_lockout = False  # ADD THIS
        self.backward_lockout_frames = 0   # ADD THIS

        # change based on speed of market
        # change based on desired buy/sell frequency
        self.buy_cooldown = 3
        self.sell_cooldown = 3

        # PRE-LOAD BUTTONS AT STARTUP
        print("🔄 Pre-loading trade buttons...")
        try:
            self.cached_buy_btn = pyautogui.locateCenterOnScreen('buy_sell/Buybutton3.png', confidence=0.8)
            print("✅ Buy button pre-loaded")
        except:
            print("❌ Buy button not found during pre-load")
            self.cached_buy_btn = None
            
        try:
            self.cached_sell_btn = pyautogui.locateCenterOnScreen('buy_sell/Sellbutton3.png', confidence=0.8)
            print("✅ Sell button pre-loaded")
        except:
            print("❌ Sell button not found during pre-load")
            self.cached_sell_btn = None

    def analyze_candles_tm(self, left_img, boxes_3020, labels_3020, scores_3020,
                right_img, boxes_1510, labels_1510, scores_1510,
                mode, candle_boxes=None, candle_labels=None,
                threshold=0.93, right_sz=640):

            decision = None
            current_time = time.time()
            valid_labels = {"LH", "HL", "HH", "LL"}

            # --- Helper: get top 2 rightmost labels ---
            def get_two_rightmost(boxes, labels, scores, min_conf=0.30):
                if not boxes or not labels or not scores:
                    return None, None
                entries = [
                    (lbl, box, score)
                    for box, lbl, score in zip(boxes, labels, scores)
                    if score >= min_conf and lbl in valid_labels
                ]
                if not entries:
                    return None, None
                entries.sort(key=lambda x: x[1][0], reverse=True)
                first = entries[0]
                second = entries[1] if len(entries) > 1 else (None, None, None)
                return first, second
            
            def save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                box_1510, rightmost_lbl_1510, score_1510,
                                box_second_1510, second_lbl_1510, score_second_1510,
                                candle_boxes):
                """Save debug images regardless of trade blocking"""
                debug_3020, debug_1510 = left_img.copy(), right_img.copy()

                if box_3020:
                    x0, y0, x1, y1 = box_3020
                    cv2.rectangle(debug_3020, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(debug_3020, f"{rightmost_lbl_3020} ({score_3020:.2f})",
                                (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if box_1510 and rightmost_lbl_1510:
                    x0, y0, x1, y1 = box_1510
                    cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(debug_1510, f"{rightmost_lbl_1510} ({score_1510:.2f})",
                                (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if box_second_1510 and second_lbl_1510:
                    x0, y0, x1, y1 = box_second_1510
                    cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(debug_1510, f"{second_lbl_1510} ({score_second_1510:.2f})",
                                (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                os.makedirs("dummy", exist_ok=True)
                cv2.imwrite("dummy/debug_3020.png", debug_3020)
                cv2.imwrite("dummy/debug_1510.png", debug_1510)

            # Get rightmost for 3020
            rightmost_lbl_3020, box_3020, score_3020 = self.get_rightmost_label(
                boxes_3020, labels_3020, scores_3020, min_conf=0.30
            )
            if rightmost_lbl_3020 not in valid_labels:
                rightmost_lbl_3020, box_3020, score_3020 = None, None, None

            # Get rightmost and second-rightmost for 1510
            first_1510, second_1510 = get_two_rightmost(
                boxes_1510, labels_1510, scores_1510, min_conf=0.30
            )
            if first_1510:
                rightmost_lbl_1510, box_1510, score_1510 = first_1510
                current_1510_rml_x = box_1510[0]  # x0 coordinate of current RML
            else:
                rightmost_lbl_1510, box_1510, score_1510 = None, None, None
                current_1510_rml_x = None
                
            if second_1510:
                second_lbl_1510, box_second_1510, score_second_1510 = second_1510
            else:
                second_lbl_1510, box_second_1510, score_second_1510 = None, None, None

            # Define current_3020_rml_x
            current_3020_rml_x = box_3020[0] if box_3020 else None

            # === SIMPLIFIED BACKWARD MOVEMENT DETECTION ===
            # Skip first frame initialization
            if self.is_first_frame:
                self.is_first_frame = False
                if current_3020_rml_x is not None:
                    self.last_correct_3020_rml_x = current_3020_rml_x
                if current_1510_rml_x is not None:
                    self.last_rml_1510_x = current_1510_rml_x
                return None  # Skip trade logic on first frame

            # Handle backward lockout countdown FIRST
            if self.backward_lockout_frames > 0:
                self.backward_lockout_frames -= 1
                if self.backward_lockout_frames == 0:
                    self.rml_backward_lockout = False
                    print("Backward movement lockout expired")
                else:
                    # Still in lockout period - block all trades
                    print(f"Backward lockout active: {self.backward_lockout_frames} frames remaining")
                    return None

            # Check for significant backward movement in 3020
            if (current_3020_rml_x is not None and 
                self.last_correct_3020_rml_x is not None and
                current_3020_rml_x < self.last_correct_3020_rml_x - 10):  # 10px tolerance
                
                print(f"3020 RML moved backwards! Blocking trades. Last: {self.last_correct_3020_rml_x}, Current: {current_3020_rml_x}")
                self.backward_lockout_frames = 15  # Lockout for 15 frames
                self.rml_backward_lockout = True
                
                save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                box_1510, rightmost_lbl_1510, score_1510,
                                box_second_1510, second_lbl_1510, score_second_1510,
                                candle_boxes)
                
                # Update tracking to prevent repeated triggers
                self.last_correct_3020_rml_x = current_3020_rml_x
                return None

            # Check for significant backward movement in 1510
            if (current_1510_rml_x is not None and 
                self.last_rml_1510_x is not None and
                current_1510_rml_x < self.last_rml_1510_x - 10):  # 10px tolerance
                
                print(f"1510 RML moved backwards! Blocking trades. Last: {self.last_rml_1510_x}, Current: {current_1510_rml_x}")
                self.backward_lockout_frames = 12
                self.rml_backward_lockout = True
                
                save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                box_1510, rightmost_lbl_1510, score_1510,
                                box_second_1510, second_lbl_1510, score_second_1510,
                                candle_boxes)
                
                # Update tracking to prevent repeated triggers
                self.last_rml_1510_x = current_1510_rml_x
                return None

            # Update normal tracking (only when not in backward movement)
            if current_3020_rml_x is not None:
                self.last_correct_3020_rml_x = current_3020_rml_x
            if current_1510_rml_x is not None:
                self.last_rml_1510_x = current_1510_rml_x

            # Current labels for SRL logic
            current_rml_1510 = rightmost_lbl_1510
            current_srl_1510 = second_lbl_1510

            # 🔓 RESET SRL LOCKOUT AFTER COOLDOWN PERIOD
            if (current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout):
                self.srl_lockout_after_trade = False
                print("SRL lockout reset - RML changed")
                
            # Print detection info
            conf_3020 = f"{int(round(score_3020 * 100))}%" if score_3020 else "N/A"
            conf_1510 = f"{int(round(score_1510 * 100))}%" if score_1510 else "N/A"
            conf_1510_second = f"{int(round(score_second_1510 * 100))}%" if score_second_1510 else "N/A"

            print(f"3020 Label: {rightmost_lbl_3020 or 'None'} with confidence {conf_3020}")
            print(f"1510 Label: {rightmost_lbl_1510 or 'None'} with confidence {conf_1510}, Box: {box_1510 or 'None'}")
            print(f"1510 Second Label: {second_lbl_1510 or 'None'} with confidence {conf_1510_second}, Box: {box_second_1510 or 'None'}")
            print(f"SRL Lockout: {self.srl_lockout_after_trade}")
            print(f"Backward Lockout: {self.backward_lockout_frames > 0}")
            
            if candle_boxes:
                rightmost_candle = max(candle_boxes, key=lambda b: b[2])  # Find by rightmost x (b[2])
                print(f"Rightmost Candle Box: {rightmost_candle}")
            else:
                print("Rightmost Candle: None")
                
            # --- Trade logic ---
            def candle_center_within_box(label_box, candle_boxes):
                """ Returns True if the rightmost candle's center is within the horizontal range of the label box. """
                if not label_box or not candle_boxes:
                    return False
                
                # Get the rightmost candle
                rightmost_candle = max(candle_boxes, key=lambda b: b[2])
                
                lx0, ly0, lx1, ly1 = label_box
                cx0, cy0, cx1, cy1 = rightmost_candle
                cx_center = (cx0 + cx1) // 2
                
                return lx0 <= cx_center+1000 <= lx1

            # PRIMARY TRADE: Continuous check for candle with current RML
            primary_trade_executed = False
            
            # BUY condition: 3020=HH and 1510=HL with candle above HL
            if (
                not self.rml_backward_lockout and # Only proceed if RML hasn't moved backwards
                rightmost_lbl_3020 == "HH"
                and rightmost_lbl_1510 == "HL"
                and mode in ("buy", "both")
                and current_time - self.last_buy_time >= self.buy_cooldown
                and candle_center_within_box(box_1510, candle_boxes)
            ):
                self.last_buy_time = current_time
                self.buy_count += 1
                decision = "BUY"
                primary_trade_executed = True
                self.srl_lockout_after_trade = True
                self.pending_srl_trade = None  
                try:
                    buy_btn = self.cached_buy_btn or pyautogui.locateCenterOnScreen(
                        'buy_sell/Buybutton3.png', confidence=0.8)
                    if buy_btn:
                        self.cached_buy_btn = buy_btn
                        pyautogui.click(buy_btn)
                    print("BUY executed - HH + HL with candle above HL")
                except pyautogui.ImageNotFoundException:
                    print("Buy button not found")

            # SELL condition: 3020=LL and 1510=LH with candle below LH
            elif (
                not self.rml_backward_lockout and
                rightmost_lbl_3020 == "LL"
                and rightmost_lbl_1510 == "LH"
                and mode in ("sell", "both")
                and current_time - self.last_sell_time >= self.sell_cooldown
                and candle_center_within_box(box_1510, candle_boxes)
            ):
                self.last_sell_time = current_time
                self.sell_count += 1
                decision = "SELL"
                primary_trade_executed = True
                self.srl_lockout_after_trade = True
                self.pending_srl_trade = None  # Clear any pending SRL trade
                try:
                    sell_btn = self.cached_sell_btn or pyautogui.locateCenterOnScreen(
                        'buy_sell/Sellbutton3.png', confidence=0.8)
                    if sell_btn:
                        self.cached_sell_btn = sell_btn
                        pyautogui.click(sell_btn)
                    print("SELL executed - LL + LH with candle below LH")
                except pyautogui.ImageNotFoundException:
                    print("Sell button not found")

            # CONTINUOUS CHECK: Update pending SRL trade when conditions are met but no candle
            if not primary_trade_executed and not self.rml_backward_lockout:
                # BUY opportunity: 3020=HH and 1510=HL but no candle
                if (
                    rightmost_lbl_3020 == "HH"
                    and rightmost_lbl_1510 == "HL"
                    and mode in ("buy", "both")
                    and current_time - self.last_buy_time >= self.buy_cooldown
                    and not candle_center_within_box(box_1510, candle_boxes)
                ):
                    # Continuously update the pending trade while conditions hold
                    self.pending_srl_trade = ("BUY", "HL")
                    print("Pending SRL BUY trade - HH + HL but no candle alignment")
                
                # SELL opportunity: 3020=LL and 1510=LH but no candle  
                elif (
                    rightmost_lbl_3020 == "LL"
                    and rightmost_lbl_1510 == "LH"
                    and mode in ("sell", "both")
                    and current_time - self.last_sell_time >= self.sell_cooldown
                    and not candle_center_within_box(box_1510, candle_boxes)
                ):
                    # Continuously update the pending trade while conditions hold
                    self.pending_srl_trade = ("SELL", "LH")
                    print("Pending SRL SELL trade - LL + LH but no candle alignment")
                    
            # SRL BACKUP TRADE: Execute when RML changes and SRL matches the pending trade
            if (not self.rml_backward_lockout and
                not self.srl_lockout_after_trade and
                current_rml_1510 != self.prev_rml_1510 and 
                current_srl_1510 != self.prev_srl_1510 and
                current_rml_1510 and current_srl_1510):  # Both are valid
                
                # Update tracking
                self.prev_rml_1510 = current_rml_1510
                self.prev_srl_1510 = current_srl_1510
                
                # Check SRL trade conditions
                if (current_srl_1510 == "HL" and 
                    rightmost_lbl_3020 == "HH" and 
                    mode in ("buy", "both") and
                    current_time - self.last_buy_time >= self.buy_cooldown):
                    
                    # Execute BUY
                    self.last_buy_time = current_time
                    self.buy_count += 1
                    decision = "BUY"
                    self.srl_lockout_after_trade = True  # 🔒 LOCK SRL TRADES after SRL trade
                    try:
                        buy_btn = self.cached_buy_btn or pyautogui.locateCenterOnScreen(
                            'buy_sell/Buybutton3.png', confidence=0.8)
                        if buy_btn:
                            self.cached_buy_btn = buy_btn
                            pyautogui.click(buy_btn)
                        print(f"BUY executed - HH + SRL {current_srl_1510} (both labels changed)")
                    except pyautogui.ImageNotFoundException:
                        print("Buy button not found")
                
                elif (current_srl_1510 == "LH" and 
                    rightmost_lbl_3020 == "LL" and 
                    mode in ("sell", "both") and
                    current_time - self.last_sell_time >= self.sell_cooldown):
                    
                    # Execute SELL
                    self.last_sell_time = current_time
                    self.sell_count += 1
                    decision = "SELL"
                    self.srl_lockout_after_trade = True  # 🔒 LOCK SRL TRADES after SRL trade
                    try:
                        sell_btn = self.cached_sell_btn or pyautogui.locateCenterOnScreen(
                            'buy_sell/Sellbutton3.png', confidence=0.8)
                        if sell_btn:
                            self.cached_sell_btn = sell_btn
                            pyautogui.click(sell_btn)
                        print(f"SELL executed - LL + SRL {current_srl_1510} (both labels changed)")
                    except pyautogui.ImageNotFoundException:
                        print("Sell button not found")

            # If only RML changed but SRL stayed same → ignore (just maturing)
            elif current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout:
                self.prev_rml_1510 = current_rml_1510
                
            save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                        box_1510, rightmost_lbl_1510, score_1510,
                        box_second_1510, second_lbl_1510, score_second_1510,
                        candle_boxes)

            return decision

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

                    if torch.cuda.is_available():
                        print(f"Model device in worker: {next(self.model.model.parameters()).device}")

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
                        trim_bottom = 230           # both monitors
                        trim_right_left_img = 150   # left monitor
                        trim_top = 45               # both monitors
                        shift_right = 40            # right monitor
                        trim_right_rimg = 400       # right monitor - THIS IS WHAT YOU WANT TO ADJUST

                    elif platform.system() == "Darwin":
                        trim_right = 300
                        trim_bottom = 300
                        trim_right_left_img = 230
                        trim_top = 60
                        shift_right = 230
                        trim_right_rimg = trim_right  # or set a specific value like 400

                    # Left image crop (unchanged)
                    left_img = full[
                        trim_top : h - trim_bottom,
                        : (w // 2) - trim_right_left_img,
                        :
                    ]

                    # FIXED: Right image crop - properly trim from the right side
                    right_img = full[
                        trim_top : h - trim_bottom,
                        (w // 2 - shift_right) : (w - trim_right - trim_right_rimg),  # CHANGED THIS LINE
                        :
                    ]

                    #  Resize for model 
                    m32 = lambda v: ((v + 31) // 32) * 32
                    left_sz = (m32(left_monitor['width']), m32(left_monitor['height']))
                    right_sz = (m32(right_monitor['width']), m32(right_monitor['height']))

                    # Model predictions 
                    combined_images = [left_img, right_img]
                    all_results = self.model.predict(
                        device=0 if torch.cuda.is_available() else 'cpu',  # ← CHANGE TO INTEGER 0
                        source=combined_images,
                        verbose=False,
                        stream=False, 
                        conf=0.01,  # Use lower confidence, filter candles later
                        iou=0.15,
                        imgsz=640
                    )

                    # Split results
                    left_results = [all_results[0]]        # First image = left_img
                    right_results = [all_results[1]]       # Second image = right_img
                    candle_results = [all_results[1]]      # Reuse right_img results for candles

                    # Process results 
                    left_boxes, left_scores, left_labels, left_conf = self.process_results(left_results)
                    right_boxes, right_scores, right_labels, right_conf = self.process_results(right_results)

                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])
                    merged_left_labels = [left_labels[i] for i in keep_left]

                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])
                    merged_right_labels = [right_labels[i] for i in keep_right]
                    
                    #process candle results
                    candle_boxes, candle_scores, candle_labels, _ = self.process_results(candle_results)
                    candle_boxes = [b for i, (b, l) in enumerate(zip(candle_boxes, candle_labels)) 
                if l == "candle" and candle_scores[i] >= 0.1]

                    decision = self.analyze_candles_tm(
                        left_img, merged_left, merged_left_labels, left_conf,
                        right_img, merged_right, merged_right_labels, right_conf,
                        mode,
                        candle_boxes=candle_boxes,
                        candle_labels=candle_labels
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
        #self.model = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt')

        #AP's Laptop
        # self.model = YOLO('/Users/Owner/StockMarket/runs/detect2/train8/weights/best.pt')

        # AP's main machine
        self.model = YOLO("c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            self.model.to('cuda')
            print("Model moved to GPU")
            # Check where model actually is
            print(f"Model device: {next(self.model.model.parameters()).device}")
        else:
            print("Using CPU")

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
            total_frames=self.total_frames,
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
    
    for i in reversed(range(3)):
        print(f"Starting in {i+1}")
        time.sleep(1)

    mw = MarketWorker()
    sys.exit(mw.app.exec_())