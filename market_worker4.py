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
import keyboard

# GPU configuration
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

class DetectionWorker(QThread):
    update_left = pyqtSignal(np.ndarray, list)
    update_right = pyqtSignal(np.ndarray, list)
    finished = pyqtSignal()

    def __init__(self, model, offset_x, offset_y, width, height, total_frames, mode):
        super().__init__()
        self.model = model
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.mode = mode
        
        self._initialize_tracking_variables()
        self._initialize_cooldowns()

    def _initialize_tracking_variables(self):
        """Initialize all tracking and state variables"""
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
        
        # Label ID tracking system
        self.label_id_map = {}  # Maps (label, x_bucket) -> unique_id
        self.next_label_id = 0
        self.last_executed_pattern = None  # Stores (3020_label, 1510_label_id, trade_type)
        self.last_traded_1510_label = None  # Track the actual label string we last traded
        self.pattern_has_evolved = False  # Flag to allow re-trading after pattern change
        self.backward_recovery_frames = 0  # Cooldown after backward movement ends
        self.frames_since_last_trade = 999  # Track frames since last trade

    def _initialize_cooldowns(self):
        """Initialize trade cooldowns based on market speed"""
        # change based on speed of market
        # change based on desired buy/sell frequency
        self.buy_cooldown = 4.5
        self.sell_cooldown = 4.5

    # ================ LABEL ID TRACKING ================
    
    def get_or_create_label_id(self, label, box):
        """
        Assign stable IDs to labels based on their initial position.
        Labels that stay within ~30px of each other get the same ID.
        This allows us to track the same label even as the chart scrolls.
        """
        if not box:
            return None
        
        # Use left x-coord as identifier with tolerance for chart scrolling
        # 30px buckets means a label moving 0-29px right keeps same ID
        x_key = round(box[0] / 30) * 30
        key = (label, x_key)
        
        if key not in self.label_id_map:
            self.label_id_map[key] = self.next_label_id
            self.next_label_id += 1
            print(f"New label ID created: {label} at x~{x_key} -> ID {self.label_id_map[key]}")
            
            # Clean old entries to prevent memory buildup (keep only last 20)
            if len(self.label_id_map) > 20:
                oldest_key = min(self.label_id_map.items(), key=lambda x: x[1])[0]
                del self.label_id_map[oldest_key]
        
        return self.label_id_map[key]

    def get_pattern_signature_by_id(self, lbl_3020, lbl_1510, box_1510, is_srl=False):
        """
        Create pattern signature using label ID instead of position.
        This way, even if the chart scrolls and x-position changes,
        we still recognize it as the same label we already traded.
        """
        if not lbl_3020 or not lbl_1510 or not box_1510:
            return None
        
        # Get stable ID for this 1510 label
        label_id = self.get_or_create_label_id(lbl_1510, box_1510)
        trade_type = "SRL" if is_srl else "RML"
        
        return (lbl_3020, label_id, trade_type)

    # ================ IMAGE PROCESSING HELPERS ================
    
    def save_debug_images(self, left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                         box_1510, rightmost_lbl_1510, score_1510,
                         box_second_1510, second_lbl_1510, score_second_1510,
                         candle_boxes):
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
            self._draw_candles_on_image(debug_1510, candle_boxes, box_1510)
        
        os.makedirs("dummy", exist_ok=True)
        cv2.imwrite("dummy/debug_3020.png", debug_3020)
        cv2.imwrite("dummy/debug_1510.png", debug_1510)

    def _draw_candles_on_image(self, image, candle_boxes, label_box):
        """Helper to draw candles and alignment lines on image"""
        for i, candle_box in enumerate(candle_boxes):
            cx0, cy0, cx1, cy1 = candle_box
            cv2.rectangle(image, (cx0, cy0), (cx1, cy1), (0, 255, 255), 2)
            
            # Mark the rightmost candle with special color
            rightmost_candle = max(candle_boxes, key=lambda b: b[2])
            if candle_box == rightmost_candle:
                cv2.rectangle(image, (cx0, cy0), (cx1, cy1), (255, 255, 0), 3)
                candle_center = (cx0 + cx1) // 2
                cv2.putText(image, f"RMC: {candle_center}",
                            (cx0, cy0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw candle center point
            candle_center = (cx0 + cx1) // 2
            cv2.circle(image, (candle_center, (cy0 + cy1) // 2), 3, (0, 0, 255), -1)
        
        # Draw alignment lines if we have both label and candles
        if label_box and candle_boxes:
            lx0, ly0, lx1, ly1 = label_box
            rightmost_candle = max(candle_boxes, key=lambda b: b[2])
            cx0, cy0, cx1, cy1 = rightmost_candle
            candle_center = (cx0 + cx1) // 2

            # Check and show alignment
            aligned = lx0 + self.plus_minus <= candle_center <= lx1 - self.plus_minus
            alignment_text = f"Aligned: {aligned}"
            cv2.putText(image, alignment_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if aligned else (0, 0, 255), 2)

    def get_two_rightmost(self, boxes, labels, scores, min_conf=0.30):
        """Get top 2 rightmost labels from detection results"""
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

    def get_rightmost_label(self, boxes, labels, scores, min_conf=0.30):
        """Get the rightmost valid label from detection results"""
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

    # ================ BACKWARD MOVEMENT DETECTION ================
    
    def _handle_backward_movement(self, current_3020_rml_x, current_1510_rml_x, 
                                 rightmost_lbl_1510, box_1510, score_1510):
        """Detect and handle backward movement in charts"""
        # Handle backward lockout countdown
        if self.backward_lockout_frames > 0:
            self.backward_lockout_frames -= 1
            if self.backward_lockout_frames == 0:
                self.rml_backward_lockout = False
                self.backward_recovery_frames = 3  # Add 3-frame cooldown after backward movement
                print("Backward movement lockout expired - entering recovery period")
            else:
                print(f"Backward lockout active: {self.backward_lockout_frames} frames remaining")
                return True
        
        # Handle backward recovery period (no trading, but detection continues)
        if self.backward_recovery_frames > 0:
            self.backward_recovery_frames -= 1
            if self.backward_recovery_frames == 0:
                print("Backward recovery complete - resuming normal operation")
            else:
                print(f"Backward recovery: {self.backward_recovery_frames} frames remaining")
                return True
        
        # SRML handling during backward movement
        if self.rml_backward_lockout:
            if rightmost_lbl_1510 and box_1510 and not hasattr(self, 'pending_srml'):
                self.pending_srml = (rightmost_lbl_1510, box_1510, score_1510)
                print(f"Stored pending SRML: {rightmost_lbl_1510}")
            return True

        elif (hasattr(self, 'pending_srml') and self.pending_srml and 
              rightmost_lbl_1510 != self.prev_rml_1510):
            
            stored_lbl, stored_box, stored_score = self.pending_srml
            if rightmost_lbl_1510 != stored_lbl:
                print(f"Restoring SRML from backward: {stored_lbl}")
                # Return as second label for SRL logic
                return True
            
            self.pending_srml = None

        # Check for backward movement in 3020
        if (current_3020_rml_x is not None and 
            self.last_correct_3020_rml_x is not None and
            current_3020_rml_x < self.last_correct_3020_rml_x - 25):
            
            print(f"3020 RML moved backwards! Last: {self.last_correct_3020_rml_x}, Current: {current_3020_rml_x}")
            self.backward_lockout_frames = 10
            self.rml_backward_lockout = True
            self.last_correct_3020_rml_x = current_3020_rml_x
            return True

        # Check for backward movement in 1510
        if (current_1510_rml_x is not None and 
            self.last_rml_1510_x is not None and
            current_1510_rml_x < self.last_rml_1510_x - 10):
            
            print(f"1510 RML moved backwards! Last: {self.last_rml_1510_x}, Current: {current_1510_rml_x}")
            self.backward_lockout_frames = 30
            self.rml_backward_lockout = True
            self.last_rml_1510_x = current_1510_rml_x
            return True

        return False

    # ================ PATTERN TRACKING ================
    
    def _update_pattern_tracking(self, rightmost_lbl_3020, rightmost_lbl_1510, box_1510):
        """Track pattern evolution and label changes"""
        # Increment frames since last trade
        self.frames_since_last_trade += 1
        
        # Get current label IDs for tracking changes
        current_1510_label_id = self.get_or_create_label_id(rightmost_lbl_1510, box_1510) if rightmost_lbl_1510 and box_1510 else None
        
        # Check if either 3020 label OR 1510 label has changed since last trade
        if self.last_executed_pattern is not None:
            last_3020_lbl = self.last_executed_pattern[0]
            last_1510_id = self.last_executed_pattern[1]
            
            # Pattern evolved if EITHER label changed
            if rightmost_lbl_3020 != last_3020_lbl or current_1510_label_id != last_1510_id:
                # Special case: If 1510 label STRING is the same as what we last traded
                # AND we're within 10 frames of the last trade (backward movement window)
                # then this is likely a position shift, not real evolution
                is_same_label_string = (rightmost_lbl_1510 == self.last_traded_1510_label)
                is_recent_trade = (self.frames_since_last_trade < 10)
                
                if is_same_label_string and is_recent_trade:
                    print(f"Position shift detected: Same label '{rightmost_lbl_1510}' within {self.frames_since_last_trade} frames of last trade - NOT marking as evolution")
                else:
                    if not self.pattern_has_evolved:
                        print(f"Pattern evolution detected: 3020 changed from {last_3020_lbl} to {rightmost_lbl_3020} OR 1510 ID changed from {last_1510_id} to {current_1510_label_id}")
                        self.pattern_has_evolved = True

    # ================ TRADE EXECUTION ================
    
    def _execute_buy_trade(self, pattern_sig, current_time, rightmost_lbl_1510):
        """Execute a BUY trade with pattern tracking"""
        self.pending_trade = ("BUY", current_time)
        self.last_buy_time = current_time
        self.buy_count += 1
        self.srl_lockout_after_trade = True
        self.pending_srl_trade = None
        self.last_executed_pattern = pattern_sig  # Save this pattern with label ID
        self.pattern_has_evolved = False  # Reset evolution flag after trading
        self.last_traded_1510_label = rightmost_lbl_1510  # Track the label string
        self.frames_since_last_trade = 0  # Reset frame counter
        
        pyautogui.hotkey('ctrl','b')
        print(f"BUY executed - Pattern with Label ID: {pattern_sig}")
        return "BUY"

    def _execute_sell_trade(self, pattern_sig, current_time, rightmost_lbl_1510):
        """Execute a SELL trade with pattern tracking"""
        self.pending_trade = ("SELL", current_time)
        self.last_sell_time = current_time
        self.sell_count += 1
        self.srl_lockout_after_trade = True
        self.pending_srl_trade = None
        self.last_executed_pattern = pattern_sig  # Save this pattern with label ID
        self.pattern_has_evolved = False  # Reset evolution flag after trading
        self.last_traded_1510_label = rightmost_lbl_1510  # Track the label string
        self.frames_since_last_trade = 0  # Reset frame counter
        
        pyautogui.hotkey('ctrl','m')
        print(f"SELL executed - Pattern with Label ID: {pattern_sig}")
        return "SELL"

    def _check_candle_alignment(self, box_1510, candle_boxes):
        """Check if label is aligned with rightmost candle"""
        if not box_1510 or not candle_boxes:
            return False
        
        rightmost_candle = max(candle_boxes, key=lambda b: b[2])
        candle_center = (rightmost_candle[0] + rightmost_candle[2]) // 2
        return box_1510[0] + self.plus_minus <= candle_center <= box_1510[2] - self.plus_minus

    def _evaluate_primary_trade(self, rightmost_lbl_3020, rightmost_lbl_1510, box_1510, 
                               candle_boxes, current_time, mode):
        """Evaluate primary RML trade conditions"""
        if self.rml_backward_lockout or self.pending_trade or not box_1510 or not candle_boxes:
            return None
        
        # BUY condition
        if (rightmost_lbl_3020 == "HH" and rightmost_lbl_1510 == "HL" and 
            mode in ("buy", "both") and current_time - self.last_buy_time >= self.buy_cooldown):
            
            if self._check_candle_alignment(box_1510, candle_boxes):
                pattern_sig = self.get_pattern_signature_by_id(
                    rightmost_lbl_3020, rightmost_lbl_1510, box_1510, is_srl=False
                )
                
                # Check duplicate pattern
                if pattern_sig == self.last_executed_pattern and not self.pattern_has_evolved:
                    print(f"DUPLICATE PATTERN BLOCKED (same pattern, no evolution): {pattern_sig}")
                    return None
                
                # Execute trade
                if pattern_sig == self.last_executed_pattern and self.pattern_has_evolved:
                    print(f"Re-trading same pattern after evolution: {pattern_sig}")
                
                return self._execute_buy_trade(pattern_sig, current_time, rightmost_lbl_1510)
        
        # SELL condition
        elif (rightmost_lbl_3020 == "LL" and rightmost_lbl_1510 == "LH" and 
              mode in ("sell", "both") and current_time - self.last_sell_time >= self.sell_cooldown):
            
            if self._check_candle_alignment(box_1510, candle_boxes):
                pattern_sig = self.get_pattern_signature_by_id(
                    rightmost_lbl_3020, rightmost_lbl_1510, box_1510, is_srl=False
                )
                
                # Check duplicate pattern
                if pattern_sig == self.last_executed_pattern and not self.pattern_has_evolved:
                    print(f"DUPLICATE PATTERN BLOCKED (same pattern, no evolution): {pattern_sig}")
                    return None
                
                # Execute trade
                if pattern_sig == self.last_executed_pattern and self.pattern_has_evolved:
                    print(f"Re-trading same pattern after evolution: {pattern_sig}")
                
                return self._execute_sell_trade(pattern_sig, current_time, rightmost_lbl_1510)
        
        return None

    def _evaluate_srl_trade(self, rightmost_lbl_3020, current_srl_1510, box_second_1510,
                           current_time, mode, current_rml_1510):
        """Evaluate SRL backup trade conditions"""
        if (self.rml_backward_lockout or self.pending_trade or 
            self.srl_lockout_after_trade or
            current_rml_1510 == self.prev_rml_1510 or 
            current_srl_1510 == self.prev_srl_1510 or
            not current_rml_1510 or not current_srl_1510):
            return None

        if not hasattr(self, 'rml_genuinely_changed') or not self.rml_genuinely_changed:
            print(f"⚠️ SRML blocked: RML didn't genuinely change")
            return None

        # CRITICAL: Check SRML candle alignment
        if not self._check_candle_alignment(box_second_1510, candle_boxes):
            print(f"⚠️ SRML blocked: No candle alignment for SRML")
            return None
            
        # Update tracking
        self.prev_rml_1510 = current_rml_1510
        self.prev_srl_1510 = current_srl_1510
        
        # SRL BUY condition
        if (current_srl_1510 == "HL" and rightmost_lbl_3020 == "HH" and 
            mode in ("buy", "both") and current_time - self.last_buy_time >= self.buy_cooldown):
            
            pattern_sig = self.get_pattern_signature_by_id(
                rightmost_lbl_3020, current_srl_1510, box_second_1510, is_srl=True
            )
            
            # Check duplicate pattern
            if pattern_sig == self.last_executed_pattern and not self.pattern_has_evolved:
                print(f"DUPLICATE SRL PATTERN BLOCKED (same pattern, no evolution): {pattern_sig}")
                return None
            
            # Execute trade
            if pattern_sig == self.last_executed_pattern and self.pattern_has_evolved:
                print(f"Re-trading same SRL pattern after evolution: {pattern_sig}")
            
            return self._execute_buy_trade(pattern_sig, current_time, current_srl_1510)
        
        # SRL SELL condition
        elif (current_srl_1510 == "LH" and rightmost_lbl_3020 == "LL" and 
              mode in ("sell", "both") and current_time - self.last_sell_time >= self.sell_cooldown):
            
            pattern_sig = self.get_pattern_signature_by_id(
                rightmost_lbl_3020, current_srl_1510, box_second_1510, is_srl=True
            )
            
            # Check duplicate pattern
            if pattern_sig == self.last_executed_pattern and not self.pattern_has_evolved:
                print(f"DUPLICATE SRL PATTERN BLOCKED (same pattern, no evolution): {pattern_sig}")
                return None
            
            # Execute trade
            if pattern_sig == self.last_executed_pattern and self.pattern_has_evolved:
                print(f"Re-trading same SRL pattern after evolution: {pattern_sig}")
            
            return self._execute_sell_trade(pattern_sig, current_time, current_srl_1510)
        
        return None

    # ================ MAIN ANALYSIS FUNCTION ================
    
    def analyze_candles_tm(self, left_img, boxes_3020, labels_3020, scores_3020,
                          right_img, boxes_1510, labels_1510, scores_1510,
                          mode, candle_boxes=None, candle_labels=None,
                          threshold=0.93, right_sz=640):
        
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
        first_1510, second_1510 = self.get_two_rightmost(
            boxes_1510, labels_1510, scores_1510, min_conf=0.30
        )
        
        # Extract first (RML) label
        if first_1510:
            rightmost_lbl_1510, box_1510, score_1510 = first_1510
            current_1510_rml_x = box_1510[0]
        else:
            rightmost_lbl_1510, box_1510, score_1510 = None, None, None
            current_1510_rml_x = None
            
        # Extract second (SRL) label
        if second_1510:
            second_lbl_1510, box_second_1510, score_second_1510 = second_1510
        else:
            second_lbl_1510, box_second_1510, score_second_1510 = None, None, None
        
        # Force SRL to None on first detection
        if not hasattr(self, 'first_rml_detected'):
            if rightmost_lbl_1510:
                self.first_rml_detected = True
                self.prev_rml_1510 = rightmost_lbl_1510
                print("First RML detected - SRL will be None until RML changes")
            second_lbl_1510 = None
            box_second_1510 = None
            score_second_1510 = None
        
        # Define current_3020_rml_x
        current_3020_rml_x = box_3020[0] if box_3020 else None

        # === FIRST FRAME INITIALIZATION ===
        if self.is_first_frame:
            self.is_first_frame = False
            if current_3020_rml_x is not None:
                self.last_correct_3020_rml_x = current_3020_rml_x
            if current_1510_rml_x is not None:
                self.last_rml_1510_x = current_1510_rml_x
            return None

        # === BACKWARD MOVEMENT DETECTION ===
        if self._handle_backward_movement(current_3020_rml_x, current_1510_rml_x,
                                         rightmost_lbl_1510, box_1510, score_1510):
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
            print("SRL lockout reset - RML changed")
            
        # Print detection info
        self._print_detection_info(rightmost_lbl_3020, score_3020, rightmost_lbl_1510, 
                                  score_1510, second_lbl_1510, score_second_1510)

        # === PATTERN TRACKING ===
        self._update_pattern_tracking(rightmost_lbl_3020, rightmost_lbl_1510, box_1510)

        # === PRIMARY TRADE: RML with candle alignment ===
        decision = self._evaluate_primary_trade(rightmost_lbl_3020, rightmost_lbl_1510, box_1510,
                                               candle_boxes, current_time, mode)
        
        if decision:
            return decision

        # === SRL BACKUP TRADE ===
        decision = self._evaluate_srl_trade(rightmost_lbl_3020, current_srl_1510, box_second_1510,
                                           current_time, mode, current_rml_1510)
        
        if decision:
            return decision

        # Update RML tracking if only RML changed (SRL stayed same - just maturing)
        elif current_rml_1510 != self.prev_rml_1510 and not self.rml_backward_lockout:
            self.prev_rml_1510 = current_rml_1510
            
        # Save debug images
        self.save_debug_images(left_img, right_img, box_3020, rightmost_lbl_3020, score_3020,
                              box_1510, rightmost_lbl_1510, score_1510,
                              box_second_1510, second_lbl_1510, score_second_1510,
                              candle_boxes)

        return decision

    def _print_detection_info(self, lbl_3020, score_3020, lbl_1510_rml, score_1510_rml,
                             lbl_1510_srl, score_1510_srl):
        """Print detection information for debugging"""
        conf_3020 = f"{int(round(score_3020 * 100))}%" if score_3020 else "N/A"
        conf_1510 = f"{int(round(score_1510_rml * 100))}%" if score_1510_rml else "N/A"
        conf_1510_second = f"{int(round(score_1510_srl * 100))}%" if score_1510_srl else "N/A"

        print(f"3020 Label: {lbl_3020 or 'None'} with confidence {conf_3020}")
        print(f"1510 Label: {lbl_1510_rml or 'None'} with confidence {conf_1510}")
        print(f"1510 Second Label: {lbl_1510_srl or 'None'} with confidence {conf_1510_second}")

    # ================ IMAGE CAPTURE AND PROCESSING ================
    
    def get_window_bounds(self, title):
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
                return 0, 0, 800, 600  # fallback default
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

    def _setup_key_detection(self):
        """Setup key press detection for cross-platform compatibility"""
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
        
        return locals().get('get_key')

    def _calculate_crop_parameters(self, h, w, platform_name):
        """Calculate crop parameters based on platform"""
        if platform_name == "Windows":
            return {
                'trim_right_ratio': 0.17,
                'trim_bottom_ratio': 0.14,
                'trim_right_left_img_ratio': 0.17,
                'trim_top_ratio': 0.05,
                'shift_right_ratio': 0.03,
                'trim_right_rimg_ratio': 0
            }
        elif platform_name == "Darwin":
            return {
                'trim_right_ratio': 0.18,
                'trim_bottom_ratio': 0.34,
                'trim_right_left_img_ratio': 0.35,
                'trim_top_ratio': 0.1,
                'shift_right_ratio': 0.16,
                'trim_right_rimg_ratio': 0.18
            }
        else:
            # Default values
            return {
                'trim_right_ratio': 0.17,
                'trim_bottom_ratio': 0.14,
                'trim_right_left_img_ratio': 0.17,
                'trim_top_ratio': 0.05,
                'shift_right_ratio': 0.03,
                'trim_right_rimg_ratio': 0
            }

    def _crop_images(self, full, platform_name):
        """Crop left and right images from the full screen capture"""
        h, w, _ = full.shape
        
        # Get crop parameters
        params = self._calculate_crop_parameters(h, w, platform_name)
        
        # Calculate pixel values
        trim_top = int(h * params['trim_top_ratio'])
        trim_bottom = int(h * params['trim_bottom_ratio'])
        trim_right_left_img = int(w//2 * params['trim_right_left_img_ratio'])
        shift_right = int(w * params['shift_right_ratio'])
        trim_right = int(w * params['trim_right_ratio'])
        trim_right_rimg = int(w * params['trim_right_rimg_ratio'])

        # Left image crop
        left_img = full[
            trim_top : h - trim_bottom,
            : (w // 2) - trim_right_left_img,
            :
        ]

        # Right image crop
        right_img = full[
            trim_top : h - trim_bottom,
            (w // 2 - shift_right) : (w - trim_right - trim_right_rimg),
            :
        ]
        
        return left_img, right_img

    def process_results(self, results):
        """Process YOLO detection results into boxes, scores, and labels"""
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
        """Apply non-max suppression to detection boxes"""
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
        """Merge boxes that are vertically close to each other"""
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

    def _log_program_stats(self, total_processing_time, frame_count):
        """Log program statistics when exiting"""
        minutes, seconds = divmod(total_processing_time, 60)
        current_time = datetime.now().strftime('%H:%M')
        current_date = str(date.today())

        # Format log content
        log_content = (
            f"\nTime: {current_time}  Date: {current_date}\n"
            f"Runtime: {int(minutes)} min {seconds:.2f} sec\n"
            f"Average runtime per frame: {total_processing_time/frame_count:.2f} seconds\n"
            f"Final number of buys: {self.buy_count}\n"
            f"Final number of sells: {self.sell_count}\n"
        )

        print(log_content)

        with open("log.txt", "a") as log_file:
            log_file.write(log_content)

    def run(self):
        """Main run loop for detection worker"""
        get_key = self._setup_key_detection()
        total_processing_time = 0

        with mss.mss() as sct:
            try:
                while self.running:
                    start_time = time.time()

                    # Detect app window dynamically
                    if platform.system() == "Darwin":
                        self.offset_x, self.offset_y, self.width, self.height = self.get_window_bounds("QuickTime Player")
                    else:
                        bounds = self.get_window_bounds("NinjaTrader 8")
                        if bounds:
                            self.offset_x, self.offset_y, self.width, self.height = bounds
                    
                    # Capture full screen
                    full = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
                    
                    # Crop images
                    left_img, right_img = self._crop_images(full, platform.system())
                    
                    # Model predictions
                    if platform.system() == "Windows":
                        fcandle_conf = 0.35
                    else:
                        fcandle_conf = 0.01

                    combined_images = [left_img, right_img]
                    all_results = self.model.predict(
                        source=combined_images,
                        verbose=False,
                        stream=False, 
                        conf=fcandle_conf,
                        iou=0.15,
                        imgsz=640,
                        device=device
                    )

                    # Split results
                    left_results = [all_results[0]]
                    right_results = [all_results[1]]
                    candle_results = [all_results[1]]

                    # Process results
                    left_boxes, left_scores, left_labels, left_conf = self.process_results(left_results)
                    right_boxes, right_scores, right_labels, right_conf = self.process_results(right_results)

                    # Apply NMS and merging
                    keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
                    merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])
                    merged_left_labels = [left_labels[i] for i in keep_left]

                    keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
                    merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])
                    merged_right_labels = [right_labels[i] for i in keep_right]
                    
                    # Process candle results
                    if platform.system() == "Darwin":
                        scandle_conf = 0.1
                    else:
                        scandle_conf = 0.4
                    
                    candle_boxes, candle_scores, candle_labels, _ = self.process_results(candle_results)
                    candle_boxes = [b for i, (b, l) in enumerate(zip(candle_boxes, candle_labels)) 
                                    if l == "candle" and candle_scores[i] >= scandle_conf]

                    # Analyze and make trade decision
                    decision = self.analyze_candles_tm(
                        left_img, merged_left, merged_left_labels, left_conf,
                        right_img, merged_right, merged_right_labels, right_conf,
                        self.mode,
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

                    print(f"\nFrame {self.frame_count} processed in {frame_processing_time:.2f} sec.")
                    time.sleep(0.0001)

                    # Stop program on 'q' key press
                    key = get_key()
                    if key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        self._log_program_stats(total_processing_time, self.frame_count)
                        break

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting...")
            finally:
                self.finished.emit()

class MarketWorker:
    def __init__(self, mode):
        self.mode = mode
        self.model = self._load_model()
        self._setup_application()

    def _load_model(self):
        """Load YOLO model based on platform"""
        if platform.system() == "Darwin":
            # Ryan's Laptop
            model_path = '/Users/ryanabbas/Desktop/work/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt'
        else:
            # AP's main machine
            model_path = "c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt"
        
        model = YOLO(model_path)
        
        if torch.cuda.is_available():
            model.to('cuda')
        elif torch.backends.mps.is_available():
            model.to('mps')
            
        return model

    def _setup_application(self):
        """Setup Qt application and detection thread"""
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
            mode=self.mode
        )

        self.detection_thread.finished.connect(self.on_finished)
        self.detection_thread.start()

    def on_finished(self):
        """Handle thread completion"""
        print("Detection finished.")
        self.app.quit()

def get_trading_mode():
    """Get trading mode from user input"""
    mode = input("Enter mode (buy / sell / both): ").strip().lower()
    while mode not in ("buy", "sell", "both"):
        mode = input("Invalid input, enter buy, sell, or both: ").strip().lower()
    return mode

if __name__ == "__main__":
    mode = get_trading_mode()
    mw = MarketWorker(mode)
    sys.exit(mw.app.exec_())

#22:39 - buys at srmm1510
#19:53 yellow label