import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from replica_screen import ReplicaScreen
import mss
import cv2
import pytesseract
import re
import os
import pyautogui
import glob
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
        self.sct = mss.mss()
        self.running = True
        self.prev_hhll_label   = None   # last HH or LL actually traded
        self.prev_hhll_box_id  = None   # (x0,y0,x1,y1) of that candle
        self.prev_trade_sig = None          # will hold ((lbl3020, box3020), (lbl1510, box1510))
        self.counter = 0 #can't sell before buying
        self.buy_count = 0 
        self.sell_count = 0
        #for debugging on gui screens
        # self.update_left.emit(debug_left, [])
        # self.update_right.emit(debug_right, [])
        
        self.templates = {}
        for lbl in ("HH", "LL", "HL", "LH"):
            template_files = glob.glob(f"templates/{lbl}/*.png")
            if not template_files:
                raise FileNotFoundError(f"No template images found in templates/{lbl}/")
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
    
    def scan_rightmost_candle(self, img, boxes, want_labels, debug_img, label_side, threshold=0.93, w_crop=130, h_crop=100):
        img_h, img_w = img.shape[:2]

        if len(boxes) == 0:
            print(f"{label_side}: No candles detected.")
            return (None, None), (None, None), debug_img

        if len(boxes) == 1:
            curr_box = boxes[0]
            x0, y0, x1, y1 = curr_box

            # Use stored previous candle box for comparison if exists
            if hasattr(self, 'prev_candle_box') and self.prev_candle_box is not None:
                prev_y0 = self.prev_candle_box[1]
            else:
                prev_y0 = y0  # if no previous, default to current

            # Decide crop box above or below based on y0 comparison
            if y0 < prev_y0:
                y_top = max(0, y0 - h_crop)
                main_patch = img[y_top:y_top + h_crop, max(0, (x0 + x1)//2 - w_crop//2) : max(0, (x0 + x1)//2 - w_crop//2) + w_crop]
                cv2.rectangle(debug_img,
                            (max(0, (x0 + x1)//2 - w_crop//2), y_top),
                            (max(0, (x0 + x1)//2 - w_crop//2) + w_crop, y_top + h_crop),
                            (0, 255, 0), 2)
            else:
                y_bot = min(img_h - h_crop, y1)
                main_patch = img[y_bot:y_bot + h_crop, max(0, (x0 + x1)//2 - w_crop//2) : max(0, (x0 + x1)//2 - w_crop//2) + w_crop]
                cv2.rectangle(debug_img,
                            (max(0, (x0 + x1)//2 - w_crop//2), y_bot),
                            (max(0, (x0 + x1)//2 - w_crop//2) + w_crop, y_bot + h_crop),
                            (0, 0, 255), 2)

            main_patch_gray = cv2.cvtColor(main_patch, cv2.COLOR_BGR2GRAY)

            label_main = None
            for lbl in want_labels:
                max_val_for_label = 0
                for tmpl in self.templates[lbl]:
                    res = cv2.matchTemplate(main_patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > max_val_for_label:
                        max_val_for_label = max_val

                if max_val_for_label >= threshold:
                    print(f"{label_side} (main box single candle): matched {lbl} with confidence {max_val_for_label:.2f}")
                    label_main = lbl
                    break

            # Draw right side box unconditionally
            fixed_right_edge = img_w - 70
            right_box_x_left = x1
            right_box_x_right = fixed_right_edge

            height = 2000
            candle_mid_y = (y0 + y1) // 2
            right_box_y_top = max(0, candle_mid_y - height // 2)
            right_box_y_bottom = min(img_h, candle_mid_y + height // 2)

            if right_box_y_bottom - right_box_y_top < 100:
                right_box_y_top = max(0, right_box_y_bottom - 100)

            cv2.rectangle(debug_img,
                        (right_box_x_left, right_box_y_top),
                        (right_box_x_right, right_box_y_bottom),
                        (255, 0, 0), 2)

            right_patch = img[right_box_y_top:right_box_y_bottom, right_box_x_left:right_box_x_right]
            label_right = None
            right_box_coords = None

            if right_patch is not None and right_patch.size > 0:
                right_patch_gray = cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY)
                for lbl in want_labels:
                    max_val_for_label = 0
                    for tmpl in self.templates[lbl]:
                        res = cv2.matchTemplate(right_patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        if max_val > max_val_for_label:
                            max_val_for_label = max_val

                    if max_val_for_label >= threshold:
                        print(f"{label_side} (right box single candle): matched {lbl} with confidence {max_val_for_label:.2f}")
                        label_right = lbl
                        right_box_coords = (right_box_x_left, right_box_y_top, right_box_x_right, right_box_y_bottom)
                        break

            # Update previous candle box for next call
            self.prev_candle_box = curr_box

            return (label_main, tuple(curr_box)), (label_right, right_box_coords), debug_img

        # --- If more than 1 box ---
        prev_box, curr_box = sorted(boxes, key=lambda b: b[0])[-2:]
        x0, y0, x1, y1 = curr_box
        prev_y = prev_box[1]

        if y0 < prev_y:
            y_top = max(0, y0 - h_crop)
            main_patch = img[y_top:y_top + h_crop, max(0, (x0 + x1)//2 - w_crop//2) : max(0, (x0 + x1)//2 - w_crop//2) + w_crop]
            cv2.rectangle(debug_img,
                        (max(0, (x0 + x1)//2 - w_crop//2), y_top),
                        (max(0, (x0 + x1)//2 - w_crop//2) + w_crop, y_top + h_crop),
                        (0, 255, 0), 2)
        else:
            y_bot = min(img_h - h_crop, y1)
            main_patch = img[y_bot:y_bot + h_crop, max(0, (x0 + x1)//2 - w_crop//2) : max(0, (x0 + x1)//2 - w_crop//2) + w_crop]
            cv2.rectangle(debug_img,
                        (max(0, (x0 + x1)//2 - w_crop//2), y_bot),
                        (max(0, (x0 + x1)//2 - w_crop//2) + w_crop, y_bot + h_crop),
                        (0, 0, 255), 2)

        main_patch_gray = cv2.cvtColor(main_patch, cv2.COLOR_BGR2GRAY)

        label_main = None
        for lbl in want_labels:
            max_val_for_label = 0
            for tmpl in self.templates[lbl]:
                res = cv2.matchTemplate(main_patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > max_val_for_label:
                    max_val_for_label = max_val

            if max_val_for_label >= threshold:
                print(f"{label_side} (main box): matched {lbl} with confidence {max_val_for_label:.2f}")
                label_main = lbl
                break

        fixed_right_edge = img_w - 70
        right_box_x_left = x1
        right_box_x_right = fixed_right_edge

        height = 1500
        candle_mid_y = (y0 + y1) // 2
        right_box_y_top = max(0, candle_mid_y - height // 2)
        right_box_y_bottom = min(img_h, candle_mid_y + height // 2)

        if right_box_y_bottom - right_box_y_top < 100:
            right_box_y_top = max(0, right_box_y_bottom - 100)

        cv2.rectangle(debug_img,
                    (right_box_x_left, right_box_y_top),
                    (right_box_x_right, right_box_y_bottom),
                    (255, 0, 0), 2)

        right_patch = img[right_box_y_top:right_box_y_bottom, right_box_x_left:right_box_x_right]
        label_right = None
        right_box_coords = None

        if right_patch is not None and right_patch.size > 0:
            right_patch_gray = cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY)
            for lbl in want_labels:
                max_val_for_label = 0
                for tmpl in self.templates[lbl]:
                    res = cv2.matchTemplate(right_patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > max_val_for_label:
                        max_val_for_label = max_val

                if max_val_for_label >= threshold:
                    print(f"{label_side} (right box): matched {lbl} with confidence {max_val_for_label:.2f}")
                    label_right = lbl
                    right_box_coords = (right_box_x_left, right_box_y_top, right_box_x_right, right_box_y_bottom)
                    break

        self.prev_candle_box = curr_box

        return (label_main, tuple(curr_box)), (label_right, right_box_coords), debug_img


    def analyze_candles_tm(self, left_img, merged_left, right_img, merged_right, templates, threshold=0.93, w_crop=130, h_crop=100):
        
        trim_amount = 100

        # Get all labels and boxes from scan_rightmost_candle for left and right sides
        (lbl_3020_main, box_3020_main), (lbl_3020_right, box_3020_right), debug_left = self.scan_rightmost_candle(
            left_img, merged_left, ("HH", "LL"), left_img.copy(), "3020", threshold, w_crop, h_crop
        )
        
        # Only continue to analyze 1510 if HH or LL is found on 3020 side
        if lbl_3020_main or lbl_3020_right:
            (lbl_1510_main, box_1510_main), (lbl_1510_right, box_1510_right), debug_right = self.scan_rightmost_candle(
                right_img, merged_right, ("HL", "LH"), right_img.copy(), "1510", threshold, w_crop, h_crop
            )

        else:
            print("No HH or LL found on 3020. Skipping 1510 detection.")
            debug_left = debug_left[trim_amount:, :]
            cv2.imwrite("dummy/debug_3020.png", debug_left)
            return None

        # Trim top 50 pixels before saving debug images
        debug_left = debug_left[trim_amount:, :]
        debug_right = debug_right[trim_amount:, :]

        # create debug files
        os.makedirs("dummy", exist_ok=True)
        cv2.imwrite("dummy/debug_3020.png", debug_left)
        cv2.imwrite("dummy/debug_1510.png", debug_right)

        #Priority check on right side boxes 
        if lbl_3020_right is not None and lbl_1510_right is not None:
            if (lbl_3020_right, lbl_1510_right) == ("HH", "HL"):
                current_sig = ((lbl_3020_right, box_3020_right), (lbl_1510_right, box_1510_right))
                if current_sig != self.prev_trade_sig:
                    print(f">> BUY signal detected (right side boxes prioritized).")
                    self.prev_trade_sig = current_sig
                    self.prev_hhll_label = lbl_3020_right
                    self.prev_hhll_box_id = box_3020_right
                    self.buy_count += 1
                    self.counter += 1
                    return "BUY"
                else:
                    print("BUY skipped — duplicate signal (right side boxes)")
                    return None

            elif (lbl_3020_right, lbl_1510_right) == ("LL", "LH"):
                if self.counter > 0:
                    current_sig = ((lbl_3020_right, box_3020_right), (lbl_1510_right, box_1510_right))
                    if current_sig != self.prev_trade_sig:
                        print(f">> SELL signal detected (right side boxes prioritized).")
                        self.prev_trade_sig = current_sig
                        self.prev_hhll_label = lbl_3020_right
                        self.prev_hhll_box_id = box_3020_right
                        self.sell_count += 1
                        self.counter -= 1
                        return "SELL"
                    else:
                        print("SELL skipped — duplicate signal (right side boxes)")
                        return None
                else:
                    print("SELL failed, you have 0 buys (right side boxes)")
                    return None

        # If no right side priority buy/sell signal detected, fallback to existing combos logic:
        combos = [
            (lbl_3020_main, box_3020_main, lbl_1510_main, box_1510_main),
            (lbl_3020_main, box_3020_main, lbl_1510_right, box_1510_right),
            (lbl_3020_right, box_3020_right, lbl_1510_main, box_1510_main),
            (lbl_3020_right, box_3020_right, lbl_1510_right, box_1510_right),
        ]

        for lbl_3020, box_3020, lbl_1510, box_1510 in combos:
            # Skip if any label is None
            if lbl_3020 is None or lbl_1510 is None:
                continue

            # Skip duplicates exactly like your old logic
            if (lbl_3020, box_3020) == (self.prev_hhll_label, self.prev_hhll_box_id):
                print(f"Skipping duplicate HH/LL signal: {lbl_3020}, {box_3020}")
                continue

            # BUY logic
            if (lbl_3020, lbl_1510) == ("HH", "HL"):
                current_sig = ((lbl_3020, box_3020), (lbl_1510, box_1510))
                if current_sig != self.prev_trade_sig:
                    print(f">> BUY signal detected (3020: {lbl_3020}, 1510: {lbl_1510})")
                    self.prev_trade_sig = current_sig
                    self.prev_hhll_label = lbl_3020
                    self.prev_hhll_box_id = box_3020
                    self.buy_count += 1
                    self.counter += 1
                    return "BUY"
                else:
                    print("BUY skipped — duplicate signal")
                    return None

            # SELL logic
            elif (lbl_3020, lbl_1510) == ("LL", "LH"):
                if self.counter > 0:
                    current_sig = ((lbl_3020, box_3020), (lbl_1510, box_1510))
                    if current_sig != self.prev_trade_sig:
                        print(f">> SELL signal detected (3020: {lbl_3020}, 1510: {lbl_1510})")
                        self.prev_trade_sig = current_sig
                        self.prev_hhll_label = lbl_3020
                        self.prev_hhll_box_id = box_3020
                        self.sell_count += 1
                        self.counter -= 1
                        return "SELL"
                    else:
                        print("SELL skipped — duplicate signal")
                        return None
                else:
                    print("SELL failed, you cannot sell before buying")
                    print(f"Number of sells you can make: {self.buy_count - self.sell_count}")
                    return None

        print("No trade signals found in any combo.")
        return None

    def analyze_candles_dynamic(self, img, coords, w_crop=130, h_crop=100):
        img_h, img_w = img.shape[:2]
        patches_per_candle = []  # to store cropped patches and positions for debug
        coords_sorted = sorted(coords, key=lambda b: b[0])  # sort by x0 (left edge)

        prev_y = None
        for idx, (x0, y0, x1, y1) in enumerate(coords_sorted):
            xc = (x0 + x1) // 2
            x_left = max(0, min(img_w - w_crop, xc - w_crop // 2))

            if idx == 0:
                # First candle gets both above and below
                y_above = max(0, y0 - h_crop)
                y_below = min(img_h - h_crop, y1)
                patch_above = img[y_above:y_above + h_crop, x_left:x_left + w_crop].copy()
                patch_below = img[y_below:y_below + h_crop, x_left:x_left + w_crop].copy()
                patches_per_candle.append({
                    'candle_idx': idx,
                    'above': patch_above,
                    'below': patch_below,
                    'boxes': ((x_left, y_above, x_left + w_crop, y_above + h_crop),
                            (x_left, y_below, x_left + w_crop, y_below + h_crop))
                })
                prev_y = y0
            else:
                # For subsequent candles, compare y0 (top y)
                if y0 < prev_y:
                    # current candle visually higher → crop above only
                    y_above = max(0, y0 - h_crop)
                    patch_above = img[y_above:y_above + h_crop, x_left:x_left + w_crop].copy()
                    patches_per_candle.append({
                        'candle_idx': idx,
                        'above': patch_above,
                        'boxes': ((x_left, y_above, x_left + w_crop, y_above + h_crop),)
                    })
                else:
                    # current candle visually lower or equal → crop below only
                    y_below = min(img_h - h_crop, y1)
                    patch_below = img[y_below:y_below + h_crop, x_left:x_left + w_crop].copy()
                    patches_per_candle.append({
                        'candle_idx': idx,
                        'below': patch_below,
                        'boxes': ((x_left, y_below, x_left + w_crop, y_below + h_crop),)
                    })
                prev_y = y0

        return patches_per_candle

    def preprocess_for_ocr(patch):
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to enhance text visibility
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    
    def classify_patch(self, main_patch_gray):
        best_label = None
        best_confidence = 0
        threshold = 0.93  # Minimum confidence required

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
            box,                   # [x1, y1, x2, y2]
            width: int | None = None,
            height: int = 200,
            pad_x: int = 10,
            side: str = "above") -> np.ndarray:

        h, w = img.shape[:2]
        x1, y1, x2, y2 = box

        # --- horizontal limits 
        if width is None:
            width = (x2 - x1) + 2 * pad_x
        left = max(0, x1 - pad_x)
        right = min(w, left + width)
        left = max(0, right - width) # recompute if we clipped on the right

        # --- vertical limits 
        if side == "above":
            top = max(0, y1 - height)
            bottom = y1
        else:  # below
            top = y2
            bottom = min(h, y2 + height)
            top = bottom - height          # recompute if we clipped at bottom

        return img[top:bottom, left:right].copy()

    def run(self):
        zoom = 0.6
        while self.running:
            start_time = time.time()

            # LEFT REGION
            lw_orig = self.width // 2
            lh_orig = self.height
            lw_zoom = int(lw_orig / zoom)
            lh_zoom = int(lh_orig / zoom)
            left_monitor = {
                "top": self.offset_y - (lh_zoom - lh_orig) // 2,
                "left": (self.offset_x - (lw_zoom - lw_orig) // 2) + 20,
                "width": lw_zoom - 70,
                "height": lh_zoom
            }

            # RIGHT REGION
            rw_orig = self.width - lw_orig
            rh_orig = self.height
            rw_zoom = int(rw_orig / zoom)
            rh_zoom = int(rh_orig / zoom)
            right_monitor = {
                "top": self.offset_y - (rh_zoom - rh_orig) // 2,
                "left": (self.offset_x + lw_orig - (rw_zoom - rw_orig) // 2) + 180,
                "width": rw_zoom - 100,
                "height": rh_zoom
            }

            left_img = np.array(self.sct.grab(left_monitor))[:, :, :3]
            right_img = np.array(self.sct.grab(right_monitor))[:, :, :3]

            m32 = lambda v: ((v + 31) // 32) * 32
            left_sz = (m32(left_monitor['width']), m32(left_monitor['height']))
            right_sz = (m32(right_monitor['width']), m32(right_monitor['height']))

            left_results = self.model.predict(
                source=left_img, verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=left_sz)
            right_results = self.model.predict(
                source=right_img, verbose=False, stream=False, conf=0.01, iou=0.15, imgsz=right_sz)

            left_boxes, left_scores = self.process_results(left_results)
            right_boxes, right_scores = self.process_results(right_results)

            keep_left = self.non_max_suppression_fast(left_boxes, left_scores, iou_thresh=0.5)
            merged_left = self.merge_vertically_close_boxes([left_boxes[i] for i in keep_left])

            keep_right = self.non_max_suppression_fast(right_boxes, right_scores, iou_thresh=0.5)
            merged_right = self.merge_vertically_close_boxes([right_boxes[i] for i in keep_right])

            # Call analyze_candles_tm with all required args
            decision = self.analyze_candles_tm(left_img, merged_left, right_img, merged_right, self.templates)

            if decision:
                print(f"Trade decision: {decision}")
                print(f"Number of buys: {self.buy_count}")
                print(f"Number of sells: {self.sell_count}")
            else:
                print(f"Number of buys: {self.buy_count}")
                print(f"Number of sells: {self.sell_count}")
                
            # Draw bounding boxes on images for visualization
            left_img = self.draw_coords_only(left_img, merged_left)
            right_img = self.draw_coords_only(right_img, merged_right)

            # Emit updated images with bounding boxes
            self.update_left.emit(left_img, merged_left)
            self.update_right.emit(right_img, merged_right)

            self.frame_count += 1
            print(f"\nFrame {self.frame_count} processed in {time.time() - start_time:.2f} sec.")
            time.sleep(2)
            
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
        self.model = YOLO('/Users/koshabbas/Desktop/work/stock_market/runs/detect/train_19/weights/last.pt')
        self.app = QApplication.instance() or QApplication(sys.argv)

        self.offset_x = 100
        self.offset_y = 120
        self.width = 700
        self.height = 410

        self.total_frames = 20 * 60 * 1  # 20 minutes at 1 fps
        
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
