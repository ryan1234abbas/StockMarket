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
from concurrent.futures import ThreadPoolExecutor, as_completed

# GPU configuration
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

class ParallelInferenceRunner:
    """Handles parallel model inferences"""
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
                
    def predict_parallel(self, left_img, right_img):
        """Run inferences in parallel"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both inference tasks
            future_left = executor.submit(
                self.models['candles_labels'].predict,
                source=[left_img, right_img],
                verbose=False,
                stream=False,
                conf=0.35 if platform.system() == "Windows" else 0.01,
                iou=0.15,
                imgsz=640,
                device=self.device
            )
            
            future_yellow = executor.submit(
                self.models['yellow_labels'].predict,
                source=right_img,
                verbose=False,
                stream=False,
                conf=0.3,
                iou=0.15,
                imgsz=640,
                device=self.device
            )
            
            # Wait for both to complete
            all_results = future_left.result()
            yellow_results = future_yellow.result()
            
        return all_results, yellow_results

class DetectionWorker(QThread):
    update_left = pyqtSignal(np.ndarray, list)
    update_right = pyqtSignal(np.ndarray, list)
    finished = pyqtSignal()

    def __init__(self, model_paths, offset_x, offset_y, width, height, total_frames):
        super().__init__()
        self.inference_runner = ParallelInferenceRunner(model_paths, device)
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
        self.rml_backward_lockout = False  
        self.backward_lockout_frames = 0   
        self.pending_srml = None
        self.plus_minus = 5
        self.pending_trade = None
        self.trade_timeout = 2.0
        
        # Yellow label detection states
        self.yellow_label_active = False
        self.last_valid_rml_1510_before_yellow = None
        self.last_valid_srl_1510_before_yellow = None
        self.last_executed_pattern = None
        self.skip_first_label_after_yellow = False
        self.first_label_after_yellow_seen = None

        # Change based on speed of market and desired buy/sell frequency
        self.buy_cooldown = 4.5
        self.sell_cooldown = 4.5
        
        # Performance tracking
        self.processing_stats = {
            'total_frames': 0,
            'total_time': 0,
            'avg_inference_time': 0,
            'avg_processing_time': 0
        }

    def _get_two_rightmost(self, boxes, labels, scores, min_conf=0.30):
        """Get top 2 rightmost labels"""
        valid_labels = {"LH", "HL", "HH", "LL"}
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

    def _get_pattern_signature(self, lbl_3020, lbl_1510, x_pos, is_srl=False):
        """Create unique pattern signature for duplicate detection"""
        if not lbl_3020 or not lbl_1510 or x_pos is None:
            return None
        x_bucket = round(x_pos / 20) * 20
        trade_type = "SRL" if is_srl else "RML"
        return (lbl_3020, lbl_1510, x_bucket, trade_type)

    def _handle_yellow_label_detected(self, rightmost_lbl_1510, second_lbl_1510):
        """Handle yellow label detection - pause all trades"""
        if not self.yellow_label_active:
            print("YELLOW LABEL DETECTED - PAUSING ALL TRADES")
            self.last_valid_rml_1510_before_yellow = self.prev_rml_1510
            self.last_valid_srl_1510_before_yellow = self.prev_srl_1510
            print(f"Stored labels before yellow: RML={self.last_valid_rml_1510_before_yellow}, SRL={self.last_valid_srl_1510_before_yellow}")
        self.yellow_label_active = True

    def _handle_yellow_label_cleared(self, rightmost_lbl_1510, second_lbl_1510):
        """Handle yellow label cleared - enter skip mode"""
        print("YELLOW LABEL CLEARED")
        self.yellow_label_active = False
        self.skip_first_label_after_yellow = True
        self.first_label_after_yellow_seen = rightmost_lbl_1510
        print(f"Current (first after yellow): RML={rightmost_lbl_1510}, SRL={second_lbl_1510}")
        print(f"Before yellow: RML={self.last_valid_rml_1510_before_yellow}, SRL={self.last_valid_srl_1510_before_yellow}")
        print(f"SKIP MODE ACTIVE - Will skip trading until next RML change")
        self.prev_rml_1510 = rightmost_lbl_1510
        self.prev_srl_1510 = second_lbl_1510
        self.srl_lockout_after_trade = False

    def _handle_skip_mode(self, rightmost_lbl_1510):
        """Handle skip mode after yellow label"""
        if rightmost_lbl_1510 != self.first_label_after_yellow_seen:
            print(f"SKIP MODE ENDED - Label changed from {self.first_label_after_yellow_seen} to {rightmost_lbl_1510}")
            print("Pattern signature cleared - ready for new trades")
            self.skip_first_label_after_yellow = False
            self.first_label_after_yellow_seen = None
            self.last_executed_pattern = None
            return False
        else:
            print(f"SKIP MODE - Still on first label after yellow ({rightmost_lbl_1510}), blocking trades")
            self.prev_rml_1510 = rightmost_lbl_1510
            return True

    def _save_debug_images(self, left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                          box_1510, rightmost_lbl_1510, score_1510,
                          box_second_1510, second_lbl_1510, score_second_1510,
                          candle_boxes, yellow_detected=False):
        """Save debug images with labels AND candles"""
        debug_3020, debug_1510 = left_img.copy(), right_img.copy()

        # Draw 3020 label (green)
        if box_3020:
            x0, y0, x1, y1 = box_3020
            cv2.rectangle(debug_3020, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(debug_3020, f"{rightmost_lbl_3020} ({score_3020:.2f})",
                        (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw 1510 RML label (red)
        if box_1510 and rightmost_lbl_1510:
            x0, y0, x1, y1 = box_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(debug_1510, f"{rightmost_lbl_1510} ({score_1510:.2f})",
                        (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw 1510 SRL label (blue)
        if box_second_1510 and second_lbl_1510:
            x0, y0, x1, y1 = box_second_1510
            cv2.rectangle(debug_1510, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(debug_1510, f"{second_lbl_1510} ({score_second_1510:.2f})",
                        (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # DRAW CANDLES on 1510 image (yellow boxes)
        if candle_boxes:
            for i, candle_box in enumerate(candle_boxes):
                cx0, cy0, cx1, cy1 = candle_box
                cv2.rectangle(debug_1510, (cx0, cy0), (cx1, cy1), (0, 255, 255), 2)
                
                rightmost_candle = max(candle_boxes, key=lambda b: b[2])
                if candle_box == rightmost_candle:
                    cv2.rectangle(debug_1510, (cx0, cy0), (cx1, cy1), (255, 255, 0), 3)
                    candle_center = (cx0 + cx1) // 2
                    cv2.putText(debug_1510, f"RMC: {candle_center}",
                                (cx0, cy0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                candle_center = (cx0 + cx1) // 2
                cv2.circle(debug_1510, (candle_center, (cy0 + cy1) // 2), 3, (0, 0, 255), -1)
        
        # Draw alignment lines if we have both label and candles
        if box_1510 and candle_boxes:
            lx0, ly0, lx1, ly1 = box_1510
            rightmost_candle = max(candle_boxes, key=lambda b: b[2])
            cx0, cy0, cx1, cy1 = rightmost_candle
            candle_center = (cx0 + cx1) // 2

            aligned = lx0+self.plus_minus <= candle_center <= lx1-self.plus_minus
            alignment_text = f"Aligned: {aligned}"
            cv2.putText(debug_1510, alignment_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if aligned else (0, 0, 255), 2)
        
        # Mark if yellow label detected
        if yellow_detected:
            cv2.putText(debug_1510, "YELLOW DETECTED - TRADES PAUSED",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        os.makedirs("dummy", exist_ok=True)
        cv2.imwrite("dummy/debug_3020.png", debug_3020)
        cv2.imwrite("dummy/debug_1510.png", debug_1510)

    def analyze_candles_tm(self, left_img, boxes_3020, labels_3020, scores_3020,
                          right_img, boxes_1510, labels_1510, scores_1510,
                          mode, candle_boxes=None, candle_labels=None,
                          threshold=0.93, right_sz=640, yellow_detected=False):

        decision = None
        current_time = time.time()
        valid_labels = {"LH", "HL", "HH", "LL"}

        # Clear timeout trades
        if self.pending_trade and current_time - self.pending_trade[1] > self.trade_timeout:
            print(f"Trade timeout - clearing stuck {self.pending_trade[0]} trade")
            self.pending_trade = None

        # Get rightmost for 3020
        rightmost_lbl_3020, box_3020, score_3020 = self.get_rightmost_label(
            boxes_3020, labels_3020, scores_3020, min_conf=0.30
        )
        if rightmost_lbl_3020 not in valid_labels:
            rightmost_lbl_3020, box_3020, score_3020 = None, None, None

        # Get rightmost and second-rightmost for 1510
        first_1510, second_1510 = self._get_two_rightmost(
            boxes_1510, labels_1510, scores_1510, min_conf=0.30
        )
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

        # YELLOW LABEL HANDLING - uses passed parameter instead of checking labels_1510
        if yellow_detected:
            self._handle_yellow_label_detected(rightmost_lbl_1510, second_lbl_1510)
            self._save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                  None, None, None, None, None, None, candle_boxes, yellow_detected=True)
            return None

        # Yellow label cleared
        if not yellow_detected and self.yellow_label_active:
            self._handle_yellow_label_cleared(rightmost_lbl_1510, second_lbl_1510)
            return None

        # Skip mode handling
        if self.skip_first_label_after_yellow:
            if self._handle_skip_mode(rightmost_lbl_1510):
                self.prev_srl_1510 = second_lbl_1510
                return None
        
        # Define current_3020_rml_x
        current_3020_rml_x = box_3020[0] if box_3020 else None

        # === BACKWARD MOVEMENT DETECTION ===
        if self.is_first_frame:
            self.is_first_frame = False
            if current_3020_rml_x is not None:
                self.last_correct_3020_rml_x = current_3020_rml_x
            if current_1510_rml_x is not None:
                self.last_rml_1510_x = current_1510_rml_x
            return None

        # Handle backward lockout countdown
        if self.backward_lockout_frames > 0:
            self.backward_lockout_frames -= 1
            if self.backward_lockout_frames == 0:
                self.rml_backward_lockout = False
                self.last_executed_pattern = None
                print("Backward movement lockout expired - pattern signature cleared")
            else:
                print(f"Backward lockout active: {self.backward_lockout_frames} frames remaining")
                return None
        
        # SRML handling during backward movement
        if self.rml_backward_lockout:
            if rightmost_lbl_1510 and box_1510 and not hasattr(self, 'pending_srml'):
                self.pending_srml = (rightmost_lbl_1510, box_1510, score_1510)
                print(f"Stored pending SRML: {rightmost_lbl_1510}")
            
            second_lbl_1510 = None
            box_second_1510 = None
            score_second_1510 = None

        elif (hasattr(self, 'pending_srml') and self.pending_srml and 
              current_rml_1510 != self.prev_rml_1510):
            
            stored_lbl, stored_box, stored_score = self.pending_srml
            if rightmost_lbl_1510 != stored_lbl:
                print(f"Restoring SRML from backward: {stored_lbl}")
                second_lbl_1510 = stored_lbl
                box_second_1510 = stored_box
                score_second_1510 = stored_score
            
            self.pending_srml = None

        # Check for backward movement in 3020
        if (current_3020_rml_x is not None and 
            self.last_correct_3020_rml_x is not None and
            current_3020_rml_x < self.last_correct_3020_rml_x - 25):
            
            print(f"3020 RML moved backwards! Last: {self.last_correct_3020_rml_x}, Current: {current_3020_rml_x}")
            self.backward_lockout_frames = 10
            self.rml_backward_lockout = True
            self.last_correct_3020_rml_x = current_3020_rml_x
            return None

        # Check for backward movement in 1510
        if (current_1510_rml_x is not None and 
            self.last_rml_1510_x is not None and
            current_1510_rml_x < self.last_rml_1510_x - 10):
            
            print(f"1510 RML moved backwards! Last: {self.last_rml_1510_x}, Current: {current_1510_rml_x}")
            self.backward_lockout_frames = 5
            self.rml_backward_lockout = True
            self.last_rml_1510_x = current_1510_rml_x
            return None

        # Update normal tracking
        if current_3020_rml_x is not None:
            self.last_correct_3020_rml_x = current_3020_rml_x
        if current_1510_rml_x is not None:
            self.last_rml_1510_x = current_1510_rml_x

        # Current labels for SRL logic
        current_rml_1510 = rightmost_lbl_1510
        current_srl_1510 = second_lbl_1510

        # RESET SRL LOCKOUT when RML changes (pattern evolved)
        if (current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout):
            self.srl_lockout_after_trade = False
            self.last_executed_pattern = None
            print("SRL lockout reset - RML changed - pattern signature cleared")
            
        # Print detection info
        conf_3020 = f"{int(round(score_3020 * 100))}%" if score_3020 else "N/A"
        conf_1510 = f"{int(round(score_1510 * 100))}%" if score_1510 else "N/A"
        conf_1510_second = f"{int(round(score_second_1510 * 100))}%" if score_second_1510 else "N/A"

        print(f"3020 Label: {rightmost_lbl_3020 or 'None'} with confidence {conf_3020}")
        print(f"1510 Label: {rightmost_lbl_1510 or 'None'} with confidence {conf_1510}, Box: {box_1510 or 'None'}")
        print(f"1510 Second Label: {second_lbl_1510 or 'None'} with confidence {conf_1510_second}, Box: {box_second_1510 or 'None'}")

        if not candle_boxes:
            print("Rightmost Candle: None")
            
        # === PRIMARY TRADE: RML with candle alignment ===
        
        # BUY condition with pattern duplicate check
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
                pattern_sig = self._get_pattern_signature(rightmost_lbl_3020, rightmost_lbl_1510, current_1510_rml_x, is_srl=False)
                
                if pattern_sig == self.last_executed_pattern:
                    print(f"DUPLICATE PATTERN BLOCKED: {pattern_sig}")
                else:
                    self.pending_trade = ("BUY", current_time)
                    self.last_buy_time = current_time
                    self.buy_count += 1
                    decision = "BUY"
                    self.srl_lockout_after_trade = True
                    self.pending_srl_trade = None
                    self.last_executed_pattern = pattern_sig
                    
                    pyautogui.hotkey('ctrl','b')
                    print(f"BUY executed - Pattern: {pattern_sig}")
                    
                    self._save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                          box_1510, rightmost_lbl_1510, score_1510,
                                          box_second_1510, second_lbl_1510, score_second_1510,
                                          candle_boxes, yellow_detected)
                    return decision

        # SELL condition with pattern duplicate check
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
                pattern_sig = self._get_pattern_signature(rightmost_lbl_3020, rightmost_lbl_1510, current_1510_rml_x, is_srl=False)
                
                if pattern_sig == self.last_executed_pattern:
                    print(f"DUPLICATE PATTERN BLOCKED: {pattern_sig}")
                else:
                    self.pending_trade = ("SELL", current_time)
                    self.last_sell_time = current_time
                    self.sell_count += 1
                    decision = "SELL"
                    self.srl_lockout_after_trade = True
                    self.pending_srl_trade = None
                    self.last_executed_pattern = pattern_sig
                    
                    pyautogui.hotkey('ctrl','m')
                    print(f"SELL executed - Pattern: {pattern_sig}")
                    
                    self._save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                          box_1510, rightmost_lbl_1510, score_1510,
                                          box_second_1510, second_lbl_1510, score_second_1510,
                                          candle_boxes, yellow_detected)
                    return decision

        # === SRL BACKUP TRADE with pattern duplicate check ===
        if (not self.rml_backward_lockout and
            not self.pending_trade and
            not self.srl_lockout_after_trade and
            current_rml_1510 != self.prev_rml_1510 and 
            current_srl_1510 != self.prev_srl_1510 and
            current_rml_1510 and current_srl_1510):
            
            self.prev_rml_1510 = current_rml_1510
            self.prev_srl_1510 = current_srl_1510
            
            # SRL BUY condition
            if (current_srl_1510 == "HL" and 
                rightmost_lbl_3020 == "HH" and 
                mode in ("buy", "both") and
                current_time - self.last_buy_time >= self.buy_cooldown):
                
                srl_x = box_second_1510[0] if box_second_1510 else None
                pattern_sig = self._get_pattern_signature(rightmost_lbl_3020, current_srl_1510, srl_x, is_srl=True)
                
                if pattern_sig == self.last_executed_pattern:
                    print(f"DUPLICATE SRL PATTERN BLOCKED: {pattern_sig}")
                else:
                    self.pending_trade = ("BUY", current_time)
                    self.last_buy_time = current_time
                    self.buy_count += 1
                    decision = "BUY"
                    self.srl_lockout_after_trade = True
                    self.last_executed_pattern = pattern_sig
                    
                    pyautogui.hotkey('ctrl','b')
                    print(f"BUY executed - SRL Pattern: {pattern_sig}")
                    
                    self._save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                          box_1510, rightmost_lbl_1510, score_1510,
                                          box_second_1510, second_lbl_1510, score_second_1510,
                                          candle_boxes, yellow_detected)
                    return decision

            # SRL SELL condition
            elif (current_srl_1510 == "LH" and 
                  rightmost_lbl_3020 == "LL" and 
                  mode in ("sell", "both") and
                  current_time - self.last_sell_time >= self.sell_cooldown):
                
                srl_x = box_second_1510[0] if box_second_1510 else None
                pattern_sig = self._get_pattern_signature(rightmost_lbl_3020, current_srl_1510, srl_x, is_srl=True)
                
                if pattern_sig == self.last_executed_pattern:
                    print(f"DUPLICATE SRL PATTERN BLOCKED: {pattern_sig}")
                else:
                    self.pending_trade = ("SELL", current_time)
                    self.last_sell_time = current_time
                    self.sell_count += 1
                    decision = "SELL"
                    self.srl_lockout_after_trade = True
                    self.last_executed_pattern = pattern_sig
                
                    pyautogui.hotkey('ctrl','m')
                    print(f"SELL executed - SRL Pattern: {pattern_sig}")
                    
                    self._save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                                          box_1510, rightmost_lbl_1510, score_1510,
                                          box_second_1510, second_lbl_1510, score_second_1510,
                                          candle_boxes, yellow_detected)
                    return decision

        # Update RML tracking if only RML changed (SRL stayed same - just maturing)
        elif current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout:
            self.prev_rml_1510 = current_rml_1510

        # Update SRL tracking
        if current_srl_1510 != self.prev_srl_1510:
            self.prev_srl_1510 = current_srl_1510
            
        self._save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                              box_1510, rightmost_lbl_1510, score_1510,
                              box_second_1510, second_lbl_1510, score_second_1510,
                              candle_boxes, yellow_detected)
        
        return decision

    def get_rightmost_label(self, boxes, labels, scores, min_conf=0.30):
        valid_labels = {"HH", "LL", "HL", "LH"}

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
                    h, w, _ = full.shape

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

                    # Run parallel inference
                    inference_start = time.time()
                    all_results, yellow_results = self.inference_runner.predict_parallel(left_img, right_img)
                    inference_time = time.time() - inference_start
                    
                    self.processing_stats['avg_inference_time'] = (
                        self.processing_stats['avg_inference_time'] * self.processing_stats['total_frames'] + inference_time
                    ) / (self.processing_stats['total_frames'] + 1)

                    # Process results
                    left_results = [all_results[0]]
                    right_results = [all_results[1]]
                    candle_results = [all_results[1]]

                    left_boxes, left_scores, left_labels, left_conf = self.process_results(left_results)
                    right_boxes, right_scores, right_labels, right_conf = self.process_results(right_results)
                    
                    # Process yellow results separately - THIS IS THE KEY FIX
                    yellow_boxes, yellow_scores, yellow_labels, _ = self.process_results(yellow_results)
                    yellow_detected = "yellow_label" in yellow_labels
                    
                    if yellow_detected:
                        print(f"Yellow label detected with confidence: {max([s for l, s in zip(yellow_labels, yellow_scores) if l == 'yellow_label'], default=0):.2f}")

                    # NMS and merging
                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])
                    merged_left_labels = [left_labels[i] for i in keep_left]

                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])
                    merged_right_labels = [right_labels[i] for i in keep_right]

                    # REMOVED: No longer appending yellow_label to merged_right_labels
                    # This was causing the duplicate trade bug

                    # Process candle results
                    scandle_conf = 0.45 if platform.system() == "Windows" else 0.1
                    candle_boxes, candle_scores, candle_labels, _ = self.process_results(candle_results)
                    candle_boxes = [b for i, (b, l) in enumerate(zip(candle_boxes, candle_labels)) 
                                   if l == "candle" and candle_scores[i] >= scandle_conf]

                    # Pass yellow_detected explicitly to analysis
                    decision = self.analyze_candles_tm(
                        left_img, merged_left, merged_left_labels, left_conf,
                        right_img, merged_right, merged_right_labels, right_conf,
                        mode,
                        candle_boxes=candle_boxes,
                        candle_labels=candle_labels,
                        yellow_detected=yellow_detected  # KEY FIX: Pass as parameter
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
                    self.processing_stats['avg_processing_time'] = self.processing_stats['total_time'] / self.processing_stats['total_frames']

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
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410
        self.total_frames = 20 * 60 * 1
        
        if platform.system() == "Darwin":
            model_paths = {
                'candles_labels': '/Users/ryanabbas/Desktop/work/StockMarket/yolo_models/candles_labels/weights/best.pt',
                'yellow_labels': '/Users/ryanabbas/Desktop/work/StockMarket/yolo_models/yellow_labels/weights/best.pt'
            }
        else:
            model_paths = {
                'candles_labels': 'c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/yolo_models/candles_labels/weights/best.pt',
                'yellow_labels': 'c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/yolo_models/yellow_labels/weights/best.pt'
            }
        
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