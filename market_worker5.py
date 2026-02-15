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

class ModelRunner:
    """Handles model inferences with conditional yellow model activation"""
    def __init__(self, model_paths, device):
        self.models = {}
        self.device = device
        self._load_models(model_paths)
        
    def _load_models(self, model_paths):
        """Load all models at once"""
        for name, path in model_paths.items():
            self.models[name] = YOLO(path)
            if torch.cuda.is_available():
                self.models[name].to('cuda')
            elif torch.backends.mps.is_available():
                self.models[name].to('mps')
                
    def predict_candles_only(self, left_img, right_img):
        """Run only candles_labels model (default mode)"""
        all_results = self.models['candles_labels'].predict(
            source=[left_img, right_img],
            verbose=False,
            stream=False,
            conf=0.35 if platform.system() == "Windows" else 0.01,
            iou=0.15,
            imgsz=640,
            device=self.device
        )
        return all_results, None
    
    def predict_with_yellow(self, left_img, right_img):
        """Run both models when yellow detection is needed"""
        all_results = self.models['candles_labels'].predict(
            source=[left_img, right_img],
            verbose=False,
            stream=False,
            conf=0.35 if platform.system() == "Windows" else 0.01,
            iou=0.15,
            imgsz=640,
            device=self.device
        )
        
        # Run yellow detection on BOTH images
        yellow_results_left = self.models['yellow_labels'].predict(
            source=left_img,
            verbose=False,
            stream=False,
            conf=0.1,
            iou=0.15,
            imgsz=640,
            device=self.device
        )
        
        yellow_results_right = self.models['yellow_labels'].predict(
            source=right_img,
            verbose=False,
            stream=False,
            conf=0.1,
            iou=0.15,
            imgsz=640,
            device=self.device
        )
        
        return all_results, (yellow_results_left, yellow_results_right)

class DetectionWorker(QThread):
    update_left = pyqtSignal(np.ndarray, list)
    update_right = pyqtSignal(np.ndarray, list)
    finished = pyqtSignal()

    def __init__(self, model_paths, offset_x, offset_y, width, height, total_frames):
        super().__init__()
        self.model_runner = ModelRunner(model_paths, device)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.frame_count = 0
        self.running = True
        self.buy_count = 0
        self.sell_count = 0
        self.last_buy_time = 0
        self.last_sell_time = 0
        self.mode = mode
        self.prev_rml_1510 = None
        self.prev_srl_1510 = None
        self.srl_lockout_after_trade = False
        self.last_correct_3020_rml_x = None
        self.is_first_frame = True
        self.last_rml_1510_x = None
        self.rml_backward_lockout = False  
        self.backward_lockout_frames = 0   
        self.pending_srml = None
        self.plus_minus = 5
        self.pending_trade = None
        self.trade_timeout = 2.0
        
        # Yellow label detection states
        self.yellow_label_active = False
        self.yellow_model_active = False  # NEW: Controls when yellow model runs
        self.last_valid_rml_1510_before_yellow = None
        self.last_valid_srl_1510_before_yellow = None
        self.last_executed_pattern = None
        self.skip_first_label_after_yellow = False
        self.first_label_after_yellow_seen = None

        # Cooldowns
        self.buy_cooldown = 4.5
        self.sell_cooldown = 4.5
        
        # Performance tracking
        self.processing_stats = {
            'total_frames': 0,
            'total_time': 0,
            'avg_inference_time': 0,
        }

    def analyze_candles_tm(self, left_img, boxes_3020, labels_3020, scores_3020,
                          right_img, boxes_1510, labels_1510, scores_1510,
                          mode, candle_boxes=None, candle_labels=None,
                          threshold=0.93, right_sz=640):

        decision = None
        current_time = time.time()
        valid_labels = {"LH", "HL", "HH", "LL"}

        # Helper: get top 2 rightmost labels (INCLUDING yellow)
        def get_two_rightmost(boxes, labels, scores, min_conf=0.30):
            if not boxes or not labels or not scores:
                return None, None
            
            valid_labels_set = {"LH", "HL", "HH", "LL", "yellow_label"}
            
            # DEBUG: Check alignment
            if len(boxes) != len(labels) or len(boxes) != len(scores):
                print(f"  MISMATCH: boxes={len(boxes)}, labels={len(labels)}, scores={len(scores)}")
            
            entries = [
                (lbl, box, score)
                for box, lbl, score in zip(boxes, labels, scores)
                if score >= min_conf and lbl in valid_labels_set
            ]
            if not entries:
                return None, None
            entries.sort(key=lambda x: x[1][0], reverse=True)
            first = entries[0]
            second = entries[1] if len(entries) > 1 else (None, None, None)
            return first, second
        
        def get_pattern_signature(lbl_3020, lbl_1510, x_pos, is_srl=False):
            if not lbl_3020 or not lbl_1510 or x_pos is None:
                return None
            x_bucket = round(x_pos / 20) * 20
            trade_type = "SRL" if is_srl else "RML"
            return (lbl_3020, lbl_1510, x_bucket, trade_type)

        if self.pending_trade and current_time - self.pending_trade[1] > self.trade_timeout:
            self.pending_trade = None

        # Get RML 3020
        rightmost_lbl_3020, box_3020, score_3020 = self.get_rightmost_label(
            boxes_3020, labels_3020, scores_3020, min_conf=0.30
        )
        # Check if 3020 RML is yellow label
        yellow_3020 = (rightmost_lbl_3020 == "yellow_label")
        
        # Filter out yellow from 3020 if present
        if rightmost_lbl_3020 not in valid_labels:
            rightmost_lbl_3020, box_3020, score_3020 = None, None, None

        # Get RML 1510 and SRML 1510 (including yellow)
        first_1510, second_1510 = get_two_rightmost(boxes_1510, labels_1510, scores_1510, min_conf=0.30)
        
        if first_1510:
            rightmost_lbl_1510, box_1510, score_1510 = first_1510
            current_1510_rml_x = box_1510[0]
        else:
            rightmost_lbl_1510, box_1510, score_1510 = None, None, None
            current_1510_rml_x = None
            
        if second_1510:
            second_lbl_1510, box_second_1510, score_second_1510 = second_1510
        else:
            second_lbl_1510, box_second_1510, score_second_1510 = None, None, None

        print(f"3020: {rightmost_lbl_3020 or 'None'}")
        print(f"1510 RML: {rightmost_lbl_1510 or 'None'}")
        print(f"1510 SRML: {second_lbl_1510 or 'None'}")

        # Yellow detection: block if yellow is RML/SRML in 1510 OR RML in 3020
        yellow_detected = (
            (rightmost_lbl_1510 == "yellow_label") or 
            (second_lbl_1510 == "yellow_label") or
            yellow_3020
        )

        if yellow_detected:
            print("YELLOW LABEL DETECTED - BLOCKING ALL TRADES")
            
            if not self.yellow_label_active:
                self.last_valid_rml_1510_before_yellow = self.prev_rml_1510
                self.last_valid_srl_1510_before_yellow = self.prev_srl_1510
            
            self.yellow_label_active = True
            
            if rightmost_lbl_1510 != "yellow_label" and rightmost_lbl_1510:
                self.prev_rml_1510 = rightmost_lbl_1510
            if second_lbl_1510 != "yellow_label" and second_lbl_1510:
                self.prev_srl_1510 = second_lbl_1510
            
            return None

        # Yellow cleared
        if not yellow_detected and self.yellow_label_active:
            print("YELLOW CLEARED - RESUMING")
            self.yellow_label_active = False
            self.skip_first_label_after_yellow = True
            self.first_label_after_yellow_seen = rightmost_lbl_1510
            self.prev_rml_1510 = rightmost_lbl_1510
            self.prev_srl_1510 = second_lbl_1510
            
            # Deactivate yellow model
            self.yellow_model_active = False
            print("Yellow model DEACTIVATED")
            
            return None

        # Skip first label after yellow
        if self.skip_first_label_after_yellow:
            if rightmost_lbl_1510 != self.first_label_after_yellow_seen:
                self.skip_first_label_after_yellow = False
                self.first_label_after_yellow_seen = None
                self.last_executed_pattern = None
            else:
                self.prev_rml_1510 = rightmost_lbl_1510
                self.prev_srl_1510 = second_lbl_1510
                return None

        # Define current_3020_rml_x
        current_3020_rml_x = box_3020[0] if box_3020 else None
       
        # Backward movement detection
        if self.is_first_frame:
            self.is_first_frame = False
            if current_3020_rml_x is not None:
                self.last_correct_3020_rml_x = current_3020_rml_x
            if current_1510_rml_x is not None:
                self.last_rml_1510_x = current_1510_rml_x
            return None
        
        current_rml_1510 = rightmost_lbl_1510
        if self.rml_backward_lockout:
            if rightmost_lbl_1510 and box_1510 and not hasattr(self, 'pending_srml'):
                self.pending_srml = (rightmost_lbl_1510, box_1510, score_1510)
            
            second_lbl_1510 = None
            box_second_1510 = None
            score_second_1510 = None

        elif (hasattr(self, 'pending_srml') and self.pending_srml and 
              current_rml_1510 != self.prev_rml_1510):
            
            stored_lbl, stored_box, stored_score = self.pending_srml
            if rightmost_lbl_1510 != stored_lbl:
                second_lbl_1510 = stored_lbl
                box_second_1510 = stored_box
                score_second_1510 = stored_score
            
            self.pending_srml = None

        # Check backward movement 3020
        if (current_3020_rml_x is not None and 
            self.last_correct_3020_rml_x is not None and
            current_3020_rml_x < self.last_correct_3020_rml_x - 25):
            print(f"3020 BACKWARD! Activating yellow model")

            self.backward_lockout_frames = 5
            self.rml_backward_lockout = True
            self.last_correct_3020_rml_x = current_3020_rml_x
            
            # Activate yellow model
            if not self.yellow_model_active:
                self.yellow_model_active = True
                print("Yellow model ACTIVATED due to backward movement")
            
            return None

        # Check backward movement 1510
        if (current_1510_rml_x is not None and 
            self.last_rml_1510_x is not None and
            current_1510_rml_x < self.last_rml_1510_x - 10):
            print(f"1510 BACKWARD! Activating yellow model")

            self.backward_lockout_frames = 5
            self.rml_backward_lockout = True
            self.last_rml_1510_x = current_1510_rml_x
            
            # Activate yellow model
            if not self.yellow_model_active:
                self.yellow_model_active = True
                print("Yellow model ACTIVATED due to backward movement")
            
            return None
        
        # Handle backward lockout countdown
        if self.backward_lockout_frames > 0:
            self.backward_lockout_frames -= 1
            if self.backward_lockout_frames == 0:
                self.rml_backward_lockout = False
                if not self.skip_first_label_after_yellow:
                    self.last_executed_pattern = None
                # Deactivate yellow model if no actual yellow was found during lockout
                if self.yellow_model_active and not self.yellow_label_active:
                    self.yellow_model_active = False
                    print("Yellow model DEACTIVATED - lockout expired, no yellow found")
            return None

        if current_3020_rml_x is not None:
            self.last_correct_3020_rml_x = current_3020_rml_x
        if current_1510_rml_x is not None:
            self.last_rml_1510_x = current_1510_rml_x

        current_srl_1510 = second_lbl_1510

        # Reset SRL lockout when RML changes
        if (current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout):
            self.srl_lockout_after_trade = False
            if not self.skip_first_label_after_yellow:
                self.last_executed_pattern = None
        
        # === PRIMARY TRADE: RML with candle alignment ===
        
        # BUY condition
        if (not self.rml_backward_lockout and 
            not self.pending_trade and
            rightmost_lbl_3020 == "HH"
            and rightmost_lbl_1510 == "HL"
            and mode in ("buy", "both")
            and current_time - self.last_buy_time >= self.buy_cooldown
            and box_1510 and candle_boxes):
            
            rightmost_candle = max(candle_boxes, key=lambda b: b[2])
            candle_center = (rightmost_candle[0] + rightmost_candle[2]) // 2
            candle_aligned = box_1510[0] + self.plus_minus <= candle_center <= box_1510[2] - self.plus_minus
            
            if candle_aligned:
                pattern_sig = get_pattern_signature(rightmost_lbl_3020, rightmost_lbl_1510, current_1510_rml_x, is_srl=False)
                
                if pattern_sig == self.last_executed_pattern:
                    return None
                
                self.pending_trade = ("BUY", current_time)
                self.last_buy_time = current_time
                self.buy_count += 1
                decision = "BUY"
                self.srl_lockout_after_trade = True
                self.pending_srl_trade = None
                self.last_executed_pattern = pattern_sig
                
                pyautogui.hotkey('ctrl','b')
                return decision

        # SELL condition
        elif (not self.rml_backward_lockout and
            not self.pending_trade and
            rightmost_lbl_3020 == "LL"
            and rightmost_lbl_1510 == "LH"
            and mode in ("sell", "both")
            and current_time - self.last_sell_time >= self.sell_cooldown
            and box_1510 and candle_boxes):
            
            rightmost_candle = max(candle_boxes, key=lambda b: b[2])
            candle_center = (rightmost_candle[0] + rightmost_candle[2]) // 2
            candle_aligned = box_1510[0] + self.plus_minus <= candle_center <= box_1510[2] - self.plus_minus
            
            if candle_aligned:
                pattern_sig = get_pattern_signature(rightmost_lbl_3020, rightmost_lbl_1510, current_1510_rml_x, is_srl=False)
                
                if pattern_sig == self.last_executed_pattern:
                    return None
                
                self.pending_trade = ("SELL", current_time)
                self.last_sell_time = current_time
                self.sell_count += 1
                decision = "SELL"
                self.srl_lockout_after_trade = True
                self.pending_srl_trade = None
                self.last_executed_pattern = pattern_sig
                
                pyautogui.hotkey('ctrl','m')
                return decision

        # === SRL BACKUP TRADE ===
        if (not self.rml_backward_lockout and
            not self.pending_trade and
            not self.srl_lockout_after_trade and
            current_rml_1510 != self.prev_rml_1510 and 
            current_srl_1510 != self.prev_srl_1510 and
            current_rml_1510 and current_srl_1510):
            
            self.prev_rml_1510 = current_rml_1510
            self.prev_srl_1510 = current_srl_1510
            
            # SRL BUY
            if (current_srl_1510 == "HL" and 
                rightmost_lbl_3020 == "HH" and 
                mode in ("buy", "both") and
                current_time - self.last_buy_time >= self.buy_cooldown):
                
                srl_x = box_second_1510[0] if box_second_1510 else None
                pattern_sig = get_pattern_signature(rightmost_lbl_3020, current_srl_1510, srl_x, is_srl=True)
                
                if pattern_sig == self.last_executed_pattern:
                    return None
                
                self.pending_trade = ("BUY", current_time)
                self.last_buy_time = current_time
                self.buy_count += 1
                decision = "BUY"
                self.srl_lockout_after_trade = True
                self.last_executed_pattern = pattern_sig
                
                pyautogui.hotkey('ctrl','b')
                return decision

            # SRL SELL
            elif (current_srl_1510 == "LH" and 
                rightmost_lbl_3020 == "LL" and 
                mode in ("sell", "both") and
                current_time - self.last_sell_time >= self.sell_cooldown):
                
                srl_x = box_second_1510[0] if box_second_1510 else None
                pattern_sig = get_pattern_signature(rightmost_lbl_3020, current_srl_1510, srl_x, is_srl=True)
                
                if pattern_sig == self.last_executed_pattern:
                    return None
                
                self.pending_trade = ("SELL", current_time)
                self.last_sell_time = current_time
                self.sell_count += 1
                decision = "SELL"
                self.srl_lockout_after_trade = True
                self.last_executed_pattern = pattern_sig
            
                pyautogui.hotkey('ctrl','m')
                return decision

        elif current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout:
            self.prev_rml_1510 = current_rml_1510

        return decision

    def get_rightmost_label(self, boxes, labels, scores, min_conf=0.30):
        valid_labels = {"HH", "LL", "HL", "LH", "yellow_label"}  # Added yellow_label

        if not boxes or not labels or not scores:
            return None, None, None

        filtered = [
            (b, l, s) for b, l, s in zip(boxes, labels, scores)
            if l in valid_labels and s >= min_conf
        ]

        if not filtered:
            return None, None, None

        box, label, score = max(filtered, key=lambda x: x[0][0])
        return label, box, score

    def run(self):
        if os.name == "posix":
            import sys, select, tty, termios

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
            atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings))
        else:
            import msvcrt
            def get_key():
                if msvcrt.kbhit():
                    return msvcrt.getch().decode("utf-8").lower()
                return None

        total_processing_time = 0

        def get_window_bounds(title):
            system = platform.system()
            if system == "Windows":
                try:
                    import pygetwindow as gw
                    win = gw.getWindowsWithTitle(title)
                    if win:
                        w = win[0]
                        return w.left, w.top, w.width, w.height
                except Exception:
                    return 0, 0, 800, 600
            elif system == "Darwin":
                return 0, 0, 1300, 1300
            else:
                return 0, 0, 800, 600

        with mss.mss() as sct:
            try:
                while self.running:
                    start_time = time.time()

                    if platform.system() == "Darwin":
                        self.offset_x, self.offset_y, self.width, self.height = get_window_bounds("QuickTime Player")
                    else:
                        bounds = get_window_bounds("NinjaTrader 8")
                        if bounds:
                            self.offset_x, self.offset_y, self.width, self.height = bounds
                            
                    full = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
                    h,w, _ = full.shape

                    if platform.system() == "Windows":
                        trim_right_ratio = 0.17           
                        trim_bottom_ratio = 0.14          
                        trim_right_left_img_ratio = 0.17  
                        trim_top_ratio = 0.05             
                        shift_right_ratio = 0.03          
                        trim_right_rimg_ratio = 0      

                    elif platform.system() == "Darwin":
                        trim_right_ratio = 0.18
                        trim_bottom_ratio = 0.34
                        trim_right_left_img_ratio = 0.35
                        trim_top_ratio = 0.1
                        shift_right_ratio = 0.16
                        trim_right_rimg_ratio = 0.18

                    trim_top = int(h * trim_top_ratio)
                    trim_bottom = int(h * trim_bottom_ratio)
                    trim_right_left_img = int(w//2 * trim_right_left_img_ratio)
                    shift_right = int(w * shift_right_ratio)
                    trim_right = int(w * trim_right_ratio)
                    trim_right_rimg = int(w * trim_right_rimg_ratio)

                    left_img = full[
                        trim_top : h - trim_bottom,
                        : (w // 2) - trim_right_left_img,
                        :
                    ]

                    right_img = full[
                        trim_top : h - trim_bottom,
                        (w // 2 - shift_right) : (w - trim_right - trim_right_rimg),
                        :
                    ]

                    inference_start = time.time()
                    
                    # Conditional model execution
                    if self.yellow_model_active:
                        print("[YELLOW MODEL ACTIVE]")
                        all_results, yellow_results = self.model_runner.predict_with_yellow(left_img, right_img)
                    else:
                        all_results, yellow_results = self.model_runner.predict_candles_only(left_img, right_img)
                    
                    inference_time = time.time() - inference_start
                    
                    self.processing_stats['avg_inference_time'] = (
                        self.processing_stats['avg_inference_time'] * self.processing_stats['total_frames'] + inference_time
                    ) / (self.processing_stats['total_frames'] + 1)

                    left_results = [all_results[0]]
                    right_results = [all_results[1]]
                    candle_results = [all_results[1]]

                    left_boxes, left_scores, left_labels, left_conf = self.process_results(left_results)
                    right_boxes, right_scores, right_labels, right_conf = self.process_results(right_results)

                    # NMS only - NO MERGING for 1510 to keep indices aligned
                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])
                    merged_left_labels = [left_labels[i] for i in keep_left]

                    # For 1510: use NMS only, skip merging to maintain label/box/score alignment
                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    
                    # Create aligned arrays after NMS (no merging)
                    nms_right_boxes = [right_boxes[i] for i in keep_right]
                    nms_right_labels = [right_labels[i] for i in keep_right]
                    nms_right_scores = [right_scores[i] for i in keep_right]

                    # Process yellow results ONLY if yellow model was active
                    yellow_found_this_frame = False
                    if yellow_results is not None:
                        yellow_results_left, yellow_results_right = yellow_results
                        
                        # Process yellow for LEFT image (3020)
                        yellow_boxes_left, yellow_scores_left, yellow_labels_left, _ = self.process_results([yellow_results_left[0]])
                        
                        # For 3020: Only merge yellow if it would be the RML (rightmost)
                        if "yellow_label" in yellow_labels_left and yellow_boxes_left:
                            # Check if this yellow would be the rightmost label
                            all_left_boxes = merged_left + yellow_boxes_left
                            all_left_labels = merged_left_labels + yellow_labels_left
                            
                            # Find rightmost among all boxes
                            if all_left_boxes:
                                rightmost_idx = max(range(len(all_left_boxes)), key=lambda i: all_left_boxes[i][0])
                                # Only add yellow if it IS the rightmost
                                if rightmost_idx >= len(merged_left):  # Yellow is rightmost
                                    yellow_found_this_frame = True
                                    for ylbl, ybox, yscore in zip(yellow_labels_left, yellow_boxes_left, yellow_scores_left):
                                        if ylbl == "yellow_label":
                                            merged_left.append(ybox)
                                            merged_left_labels.append("yellow_label")
                                            break
                        
                        # Process yellow for RIGHT image (1510)
                        yellow_boxes_right, yellow_scores_right, yellow_labels_right, _ = self.process_results([yellow_results_right[0]])
                        
                        # For 1510: Only merge yellow if it would be RML or SRML (top 2 rightmost)
                        if "yellow_label" in yellow_labels_right and yellow_boxes_right:
                            # Check if this yellow would be in top 2 rightmost
                            all_right_boxes = nms_right_boxes + yellow_boxes_right
                            all_right_labels = nms_right_labels + yellow_labels_right
                            
                            if all_right_boxes:
                                # Get top 2 rightmost
                                sorted_indices = sorted(range(len(all_right_boxes)), key=lambda i: all_right_boxes[i][0], reverse=True)
                                top_2_indices = sorted_indices[:2] if len(sorted_indices) >= 2 else sorted_indices
                                
                                # Check if yellow is in top 2
                                yellow_idx = len(nms_right_boxes)  # First yellow index after existing boxes
                                if yellow_idx in top_2_indices:  # Yellow is RML or SRML
                                    yellow_found_this_frame = True
                                    for ylbl, ybox, yscore in zip(yellow_labels_right, yellow_boxes_right, yellow_scores_right):
                                        if ylbl == "yellow_label":
                                            nms_right_boxes.append(ybox)
                                            nms_right_labels.append("yellow_label")
                                            nms_right_scores.append(yscore)
                                            break
                    
                    # If yellow was found this frame, reset the lockout to keep model active
                    if yellow_found_this_frame and self.yellow_model_active:
                        self.backward_lockout_frames = 5  # Reset lockout to keep scanning
                    
                    scandle_conf = 0.45 if platform.system() == "Windows" else 0.1
                    
                    candle_boxes, candle_scores, candle_labels, _ = self.process_results(candle_results)
                    candle_boxes = [b for i, (b, l) in enumerate(zip(candle_boxes, candle_labels)) 
                                   if l == "candle" and candle_scores[i] >= scandle_conf]

                    decision = self.analyze_candles_tm(
                        left_img, merged_left, merged_left_labels, left_conf,
                        right_img, nms_right_boxes, nms_right_labels, nms_right_scores,
                        mode,
                        candle_boxes=candle_boxes,
                        candle_labels=candle_labels
                    )

                    if decision:
                        print(f"Trade decision: {decision}")
                    print(f"Number of buys: {self.buy_count}")
                    print(f"Number of sells: {self.sell_count}")

                    self.frame_count += 1
                    frame_processing_time = time.time() - start_time
                    total_processing_time += frame_processing_time
                    self.processing_stats['total_frames'] += 1
                    self.processing_stats['total_time'] += frame_processing_time

                    print(f"\nFrame {self.frame_count} processed in {frame_processing_time:.2f} sec.")
                    print(f"Avg inference time: {self.processing_stats['avg_inference_time']:.2f} sec")
                    time.sleep(0.0001)

                    key = get_key()
                    if key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        minutes, seconds = divmod(total_processing_time, 60)

                        current_time = datetime.now().strftime('%H:%M')
                        current_date = str(date.today())

                        log_content = (
                            f"\nTime: {current_time}  Date: {current_date}\n"
                            f"Runtime: {int(minutes)} min {seconds:.2f} sec\n"
                            f"Average runtime per frame: {total_processing_time / self.frame_count:.2f} seconds\n"
                            f"Average inference time: {self.processing_stats['avg_inference_time']:.2f} seconds\n"
                            f"Final number of buys: {self.buy_count}\n"
                            f"Final number of sells: {self.sell_count}\n"
                        )

                        print(log_content)

                        with open("log.txt", "a") as log_file:
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
                
                h, w = result.orig_shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    conf = box.conf[0].item()
                    scores.append(conf)
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
        
        if platform.system() == "Darwin":
            model_paths = {
                'candles_labels': '/Users/ryanabbas/Desktop/work/StockMarket/yolo_models/candles_labels/weights/best.pt',
                'yellow_labels': '/Users/ryanabbas/Desktop/work/StockMarket/yolo_models/yellow_labels/weights/best.pt'
            }
        else:
            model_paths = {
                'candles_labels': 'c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/yolo_models/candles_labels/weights/best.pt',
                'yellow_labels': "c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/yolo_models/yellow_labels/weights/best.pt"
            }

        self.app = QApplication.instance() or QApplication(sys.argv)
        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410
        self.total_frames = 20 * 60 * 1  
        
        self.detection_thread = DetectionWorker(
            model_paths=model_paths,
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

    mw = MarketWorker()
    sys.exit(mw.app.exec_())