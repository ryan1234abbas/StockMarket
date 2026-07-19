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
from datetime import date, datetime
import torch

# =====================================================================
# Midband_entry.py
# Signal logic (layout: 30020 LEFT, 15010 RIGHT):
#   BUY signal:  3020 rightmost label == HH  AND  1510 rightmost label == HH
#   SELL signal: 3020 rightmost label == LL  AND  1510 rightmost label == LL
# Entry is a LIMIT order at the midband line on the 1510 chart via
# NinjaTrader chart mouse order hotkey (per the user's Hot Keys settings:
# Order Entry -> Chart mouse orders -> Limit = Shift+LeftClick):
#   BUY  -> shift+click on the thick chartreuse line  (#7FFF00)
#   SELL -> shift+click on the thick magenta line     (#FF00FF)
# (Alt+LeftClick is Stop Limit in this setup - do NOT use alt.)
#
# Exit: same as worker7 - Close is clicked when the 1510 background area
# transitions against the position. NinjaTrader's Close also cancels any
# resting unfilled limit order, so an invalidated entry is cleaned up too.
# =====================================================================

pyautogui.PAUSE = 0.01

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

if platform.system() == "Darwin":
    MODEL_PATH = '/Users/ryanabbas/Desktop/work/StockMarket/yolo_models/candles_labels/weights/best.pt'
else:
    MODEL_PATH = 'c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/yolo_models/candles_labels/weights/best.pt'

# Chart plot areas as ratios of the FULL screen (x1, y1, x2, y2).
# y2 extends to 0.96 so labels printed at the very bottom of a chart
# (fresh LLs in a falling market) are not clipped out of the crop.
LEFT_CHART = (0.00, 0.02, 0.42, 0.96)    # 30020
RIGHT_CHART = (0.465, 0.02, 0.83, 0.96)  # 15010

# Close button (used for exits and to clear any resting order before a new entry)
CLOSE_BUTTON_RATIO = (0.967, 0.174)
CLOSE_BUTTON_IMG = "close.png"

# Midband line colors, BGR (user hex: buy #FF7FFF00 chartreuse, sell #FFFF00FF magenta)
BUY_LINE_BGR = (0, 255, 127)
SELL_LINE_BGR = (255, 0, 255)
LINE_TOL = 45            # per-channel color tolerance
LINE_SCAN_WIDTH = 120    # pixels left of the plot edge scanned for the line
MIN_LINE_AREA = 25       # px; picks the THICK band, ignores thin dashed lines

# The resting limit order FOLLOWS the midband line, managed by the bot
# itself (the NinjaTrader "Attach To Indicator" menu automation proved
# unfixable - menu structure changes with ATM state, keyboard focus rules
# differ per menu). When the line moves, the order is cancelled via its
# tag menu (right-click tag -> click order item -> type 'c': 'Cancel
# Order' is the only C item, so the letter activates it directly) and
# re-placed at the line's new position. Tracking stops once price bricks
# touch the order level (assumed filled).
TRACK_DELAY_AFTER_PLACE = 0.8   # wait for the order tag to render
REPOSITION_THRESHOLD_PX = 10    # move the order when the line moved this much
REPOSITION_MIN_INTERVAL = 2.0   # seconds between cancel/replace cycles
TAG_MISSING_FRAMES = 6          # tag gone this many frames -> filled/cancelled

# ===== FEATURE SWITCHES =====
# The ENTRY side (signal -> limit at the midband line) is proven and always
# on. Everything else is opt-in, OFF by default, so the bot's core behavior
# stays reliable while extras are verified one at a time:
TRACK_ORDER = False   # move the resting limit with the line (cancel/replace).
                      # Needs real-time speed - menus take ~2s per move.
AUTO_CLOSE = True     # area-color exits + close-before-entry Close clicks.
                      # close.png (cropped from a real 4K capture, verified
                      # match=1.000 across sessions) locates the button.
ORDER_MENU_MAX_HEIGHT = 170  # px; the order menu has ONE item ('Buy N @ .. Entry').
                             # A taller first menu means the right-click missed the
                             # tag and hit something else (e.g. a drawing object,
                             # whose menu contains 'Remove' - never click blindly!)
MENU_EXCLUDE_X_RATIO = 0.87  # reject candidate menus that lie ENTIRELY right of
                             # this (the DOM panel repaints constantly and mimics
                             # a popup). Menus anchored on the chart always START
                             # left of it, even when they spill over the DOM -
                             # so straddling regions are kept whole.

LABEL_CONF = 0.30
YOLO_CONF = 0.25
STABLE_FRAMES = 2
BUY_COOLDOWN = 3.0
SELL_COOLDOWN = 3.0
Y_BUCKET = 15
STATUS_EVERY_N_FRAMES = 30
CLOSE_BEFORE_ENTRY_DELAY = 0.2   # after Close click, before placing the limit

# 1510 background-area exit (same as worker7)
AREA_EDGE_WIDTH = 150
AREA_STRIP_WIDTH = 25
AREA_MIN_PIXELS = 150
AREA_DOMINANCE = 1.5
AREA_STABLE_FRAMES = 3

VALID_LABELS = {"HH", "HL", "LL", "LH"}


def extract_detections(result):
    if result.boxes is None or len(result.boxes) == 0:
        return [], [], []
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    labels = [result.names[c] for c in cls]
    return list(xyxy), labels, list(conf)


def rightmost_label(boxes, labels, scores, min_conf=LABEL_CONF):
    best = None
    for b, l, s in zip(boxes, labels, scores):
        if l in VALID_LABELS and s >= min_conf:
            if best is None or b[0] > best[1][0]:
                best = (l, b)
    return best if best else (None, None)


def evaluate_frame(left_dets, right_dets):
    """BUY when 3020 RML is HH and the 1510 RML is HH (entry at the green
    midband line), SELL when 3020 RML is LL and 1510 RML is LL (entry at
    the magenta line). Fires for EVERY NEW 1510 label: the caller resets
    the dedup signature whenever the 1510 RML changes."""
    l_lbl, l_box = rightmost_label(*left_dets)
    r_lbl, r_box = rightmost_label(*right_dets)

    debug = f"3020={l_lbl or '-'} 1510={r_lbl or '-'}"

    if l_lbl == "HH" and r_lbl == "HH":
        return "BUY", debug, r_box
    if l_lbl == "LL" and r_lbl == "LL":
        return "SELL", debug, r_box
    return None, debug, None


def plot_right_edge(crop_bgr):
    """Rightmost column of the chart PLOT area, found via the background
    shading. Everything right of this is the price axis - scanning or
    clicking there must never happen (price marker boxes on the axis are
    the same colors as the midband lines)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2HSV)
    pink1 = cv2.inRange(hsv, (0, 15, 110), (15, 95, 255))
    pink2 = cv2.inRange(hsv, (160, 15, 110), (180, 95, 255))
    sage = cv2.inRange(hsv, (35, 15, 110), (85, 95, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(pink1, pink2), sage)
    cols = (mask > 0).sum(axis=0)
    h = crop_bgr.shape[0]
    shaded = np.where(cols >= h * 0.3)[0]
    return int(shaded[-1]) + 1 if shaded.size else crop_bgr.shape[1]


def detect_line(crop_bgr, target_bgr, scan_width=LINE_SCAN_WIDTH,
                tol=LINE_TOL, min_area=MIN_LINE_AREA):
    """Find the thick colored midband line just left of the chart's plot
    edge. Returns (y, x) in crop coordinates of the line's center, or None.
    Uses the largest connected color blob so thin dashed lines of a similar
    color are ignored, and scans only INSIDE the plot area so magenta/green
    price boxes on the axis can never be mistaken for the line."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    edge = plot_right_edge(crop_bgr)
    x0 = max(0, edge - scan_width)
    if edge - x0 < 5:
        return None
    region = np.ascontiguousarray(crop_bgr[:, x0:edge, :])
    target = np.array(target_bgr, dtype=np.int16)
    lower = np.clip(target - tol, 0, 255).astype(np.uint8)
    upper = np.clip(target + tol, 0, 255).astype(np.uint8)
    mask = cv2.inRange(region, lower, upper)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_i, best_area = None, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_i, best_area = i, area
    if best_i is None:
        return None
    cx, cy = centroids[best_i]
    return int(cy), x0 + int(cx)


def find_order_tag(crop_bgr, y_hint=None):
    """Locate the cyan order tag (e.g. '3 Sell LMT') anywhere on the chart
    crop - the chart autoscales, so the tag can drift far from where the
    order was placed. Returns (y, x) of the tag center in crop coordinates,
    or None. Thin cyan trend lines are eroded away; among multiple candidate
    blobs the one closest to y_hint wins (largest otherwise)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    region = np.ascontiguousarray(crop_bgr)
    b, g, r = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    mask = ((b > 180) & (g > 180) & (r < 140)).astype(np.uint8) * 255

    # Erode away thin cyan trend lines (1-2px) that cross the tag and would
    # otherwise merge with it into one misshapen blob; the solid tag survives
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, n):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 30 <= w <= 300 and 8 <= hh <= 50 and area >= 0.5 * w * hh:
            candidates.append((area, centroids[i]))
    if not candidates:
        return None
    if y_hint is not None:
        _, (cx, cy) = min(candidates, key=lambda c: abs(c[1][1] - y_hint))
    else:
        _, (cx, cy) = max(candidates, key=lambda c: c[0])
    return int(cy), int(cx)


def find_new_menu(before_bgr, after_bgr, min_area=6000,
                  exclude_x_ratio=MENU_EXCLUDE_X_RATIO):
    """Locate a context menu that appeared between two full-screen grabs:
    the largest newly-changed region whose pixels are menu-background gray.
    Works no matter which side of the parent the submenu opens on, and chart
    repaints don't qualify (they are not large solid light-gray areas). The
    screen right of exclude_x_ratio (the DOM panel) is ignored entirely.
    Returns (x, y, w, h) in screen-grab coordinates, or None."""
    diff = cv2.absdiff(before_bgr, after_bgr).max(axis=2)
    b = after_bgr[:, :, 0].astype(np.int16)
    g = after_bgr[:, :, 1].astype(np.int16)
    r = after_bgr[:, :, 2].astype(np.int16)
    grayish = ((np.abs(b - g) < 14) & (np.abs(g - r) < 14) &
               (b > 210) & (b < 253))
    mask = ((diff > 25) & grayish).astype(np.uint8) * 255
    # Bridge the dark text inside the menu into one solid region
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    dom_x = (int(mask.shape[1] * exclude_x_ratio)
             if exclude_x_ratio is not None else None)
    best = None
    for i in range(1, n):
        left = int(stats[i, cv2.CC_STAT_LEFT])
        if dom_x is not None and left >= dom_x:
            continue  # fully inside the DOM column - a repaint, not a menu
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and (best is None or area > best[0]):
            best = (area, (left,
                           int(stats[i, cv2.CC_STAT_TOP]),
                           int(stats[i, cv2.CC_STAT_WIDTH]),
                           int(stats[i, cv2.CC_STAT_HEIGHT])))
    return best[1] if best else None


def price_touched_level(crop_bgr, line_y, band=4, min_pixels=25):
    """True if price bricks have reached the given y level at the right edge
    of the chart - used to assume the resting limit order has FILLED and
    stop repositioning it. Bricks are saturated colors whose hue is NOT one
    of the two midband line colors (chartreuse ~45, magenta ~150)."""
    if crop_bgr is None or crop_bgr.size == 0 or line_y is None:
        return False
    h = crop_bgr.shape[0]
    edge = plot_right_edge(crop_bgr)
    x0 = max(0, edge - 25)
    y0 = max(0, line_y - band)
    y1 = min(h, line_y + band + 1)
    if y1 <= y0 or edge <= x0:
        return False
    region = np.ascontiguousarray(crop_bgr[y0:y1, x0:edge, :])
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    line_hue = ((hue >= 38) & (hue <= 62)) | ((hue >= 140) & (hue <= 168))
    brick = (sat >= 120) & (val >= 60) & ~line_hue
    return int(brick.sum()) >= min_pixels


def classify_area(crop_bgr):
    """'GREEN'/'RED'/None for the background shading band at the right edge."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2HSV)
    pink1 = cv2.inRange(hsv, (0, 15, 110), (15, 95, 255))
    pink2 = cv2.inRange(hsv, (160, 15, 110), (180, 95, 255))
    pink = cv2.bitwise_or(pink1, pink2)
    sage = cv2.inRange(hsv, (35, 15, 110), (85, 95, 255))
    pink_cols = (pink > 0).sum(axis=0)
    sage_cols = (sage > 0).sum(axis=0)
    h = crop_bgr.shape[0]
    shaded = np.where((pink_cols + sage_cols) >= h * 0.3)[0]
    if shaded.size == 0:
        return None
    x_last = int(shaded[-1])
    window = slice(max(0, x_last - AREA_STRIP_WIDTH + 1), x_last + 1)
    r = int(pink_cols[window].sum())
    g = int(sage_cols[window].sum())
    if g >= AREA_MIN_PIXELS and g > r * AREA_DOMINANCE:
        return "GREEN"
    if r >= AREA_MIN_PIXELS and r > g * AREA_DOMINANCE:
        return "RED"
    return None


def save_trade_snapshot(tag, left_img, right_img, left_dets, right_dets, click_yx=None):
    os.makedirs("trade_snaps", exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    for name, img, dets in (("left", left_img, left_dets),
                            ("right", right_img, right_dets)):
        vis = img.copy()
        for b, l, s in zip(*dets):
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(vis, f"{l} {s:.2f}", (b[0], max(12, b[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if name == "right" and click_yx is not None:
            cv2.drawMarker(vis, (click_yx[1], click_yx[0]), (0, 0, 0),
                           cv2.MARKER_CROSS, 30, 3)
        cv2.imwrite(os.path.join("trade_snaps", f"{ts}_{tag}_{name}.png"), vis)


class MidbandWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, trade_mode):
        super().__init__()
        self.mode = trade_mode
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.last_buy_time = 0
        self.last_sell_time = 0
        self.last_executed_sig = None
        self.candidate_signal = None
        self.candidate_frames = 0
        self.button_pos = {}

        # Screen origins of the 1510 crop, set in run()
        self.right_origin_x = 0
        self.chart_origin_y = 0

        # Position + 1510 area-color exit tracking
        self.position = None
        self.close_count = 0
        self.area_confirmed = None
        self.area_candidate = None
        self.area_frames = 0

        # Resting-order tracking: the limit follows the midband line
        self.tracked = None
        self.last_reposition_time = 0.0

        # 1510 RML tracking: a CHANGED label re-arms the dedup so every
        # new HH/LL can trade, even at a previously traded price level
        self.prev_r_lbl = None

        print("Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)
        if device != 'cpu':
            self.model.to(device)
        self.use_half = (device == 'cuda')
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(source=[dummy, dummy], verbose=False, imgsz=640,
                           device=device, half=self.use_half, conf=YOLO_CONF, iou=0.15)
        print("Model ready.")

    def find_button(self, which):
        if which in self.button_pos:
            return self.button_pos[which]
        pos = None
        if os.path.exists(CLOSE_BUTTON_IMG):
            try:
                loc = pyautogui.locateCenterOnScreen(CLOSE_BUTTON_IMG, confidence=0.85)
                if loc:
                    pos = (int(loc.x), int(loc.y))
                    print(f"CLOSE button located by image at {pos}")
            except Exception as e:
                print(f"Template search for CLOSE failed ({e}), using ratio fallback")
        if pos is None:
            sw, sh = pyautogui.size()
            pos = (int(sw * CLOSE_BUTTON_RATIO[0]), int(sh * CLOSE_BUTTON_RATIO[1]))
            print(f"CLOSE button using ratio position {pos}")
        self.button_pos[which] = pos
        return pos

    def log_event(self, text):
        with open("midband_debug.log", "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} {text}\n")

    def close_position(self, reason):
        if not AUTO_CLOSE:
            # Auto-closing disabled: reset state so new entries are allowed,
            # but never click - exits and cancels are managed manually
            print(f"AUTO_CLOSE off ({reason}) - manage the exit manually")
            self.log_event(f"AUTO_CLOSE OFF {self.position or 'order'} - {reason}")
            self.position = None
            self.tracked = None
            return
        x, y = self.find_button("CLOSE")
        if not os.path.exists(CLOSE_BUTTON_IMG):
            # Ratio fallback in use: verify the dark Close button actually
            # sits at that spot (the DOM may be absent from the layout).
            # Chart background there is light (~180+); the button is dark.
            with mss.mss() as sct:
                probe = np.array(sct.grab({"left": x - 3, "top": y - 3,
                                           "width": 7, "height": 7,
                                           "mon": 1}))[:, :, :3]
            if probe.mean() > 110:
                print(f"CLOSE SKIPPED ({reason}) - no Close button at the "
                      "expected DOM position; manage exits manually")
                self.log_event(f"CLOSE SKIPPED {self.position or 'order'} - {reason}")
                self.position = None
                self.tracked = None
                return
        pyautogui.click(x, y)
        self.close_count += 1
        print(f"CLOSED {self.position or 'order'} - {reason}")
        self.log_event(f"CLOSE {self.position or 'order'} - {reason}")
        self.position = None
        self.tracked = None

    def click_limit_at(self, hit):
        """Cancel all working orders, then shift+click the given point on the
        1510 chart (NinjaTrader chart mouse order: Limit = Shift+LeftClick)."""
        y, x = hit
        screen_x = self.right_origin_x + x
        screen_y = self.chart_origin_y + y

        # Focus click first: the console usually holds keyboard focus, and a
        # click into an unfocused window is consumed as window activation.
        # It lands OFFSET from the order point with a pause afterwards so
        # the pair can never be interpreted as a double-click. A plain
        # left-click places no order (Limit needs shift), so it is safe.
        pyautogui.click(screen_x - 120, screen_y)
        time.sleep(0.3)
        pyautogui.keyDown('shift')
        time.sleep(0.05)
        pyautogui.click(screen_x, screen_y)
        time.sleep(0.05)
        pyautogui.keyUp('shift')
        return screen_x, screen_y

    def place_limit_at_line(self, side, right_img):
        """Place a limit order at the midband line and start tracking it.
        Returns (side, click_yx) or (None, None) if the line was not found."""
        line_bgr = BUY_LINE_BGR if side == "BUY" else SELL_LINE_BGR
        hit = detect_line(right_img, line_bgr)
        if hit is None:
            self.log_event(f"SIGNAL {side} but midband line not found - no order")
            print(f"{side} signal but midband line NOT FOUND - no order placed")
            return None, None

        # Close only when FLIPPING direction; same-direction adds keep the
        # existing position/orders (one Close flattens and cancels them all)
        opposite = "SHORT" if side == "BUY" else "LONG"
        if self.position == opposite:
            self.close_position("flipping direction")
            time.sleep(CLOSE_BEFORE_ENTRY_DELAY)

        screen_x, screen_y = self.click_limit_at(hit)

        if side == "BUY":
            self.buy_count += 1
            self.position = "LONG"
        else:
            self.sell_count += 1
            self.position = "SHORT"
        # Start tracking: the order will follow the midband line
        self.tracked = {"side": side, "y": hit[0], "time": time.time(),
                        "tag_missing": 0}
        self.last_reposition_time = time.time()
        print(f"LIMIT {side} placed at midband line (screen {screen_x},{screen_y})")
        return side, hit

    def cancel_order_via_menu(self, tag_yx):
        """Cancel the resting order through its tag menu, using ONLY the
        proven steps: right-click the tag (verify the short single-item
        order menu), click the order item, then type 'c' - 'Cancel Order'
        is the only item starting with C, so the letter activates it
        directly with no positional navigation. Returns True on success."""
        os.makedirs("rpa_debug", exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")

        def fail(step, shot=None, box=None):
            if shot is not None:
                self._save_rpa_shot(ts, f"FAIL_{step}", shot, box)
            for _ in range(3):
                pyautogui.press('esc')
                time.sleep(0.12)
            self.log_event(f"CANCEL FAILED at step: {step}")
            print(f"CANCEL FAILED at step '{step}' - menus closed")
            return False

        with mss.mss() as sct:
            mon = sct.monitors[1]

            def grab():
                return np.array(sct.grab(mon))[:, :, :3]

            sx = self.right_origin_x + tag_yx[1]
            sy = self.chart_origin_y + tag_yx[0]

            # 1) right-click the order tag -> 'Buy/Sell N @ price Entry' menu
            before = grab()
            pyautogui.rightClick(sx, sy)
            time.sleep(0.7)
            shot1 = grab()
            menu1 = find_new_menu(before, shot1)
            if menu1 is None:
                return fail("order-menu", shot1)
            # SAFETY: the order menu has exactly one item, so it is SHORT.
            # A tall menu means the right-click missed the tag and opened
            # something else (a drawing object's menu contains 'Remove' -
            # clicking blindly there is dangerous). Abort and retry.
            if menu1[3] > ORDER_MENU_MAX_HEIGHT:
                return fail("wrong-menu-not-order", shot1, menu1)
            self._save_rpa_shot(ts, "cancel_1_menu", shot1, menu1)

            # 2) click the (single) order entry item -> order submenu opens
            pyautogui.moveTo(mon["left"] + menu1[0] + menu1[2] // 2,
                             mon["top"] + menu1[1] + menu1[3] // 2)
            time.sleep(0.35)
            pyautogui.click()
            time.sleep(0.7)
            shot2 = grab()
            submenu = find_new_menu(shot1, shot2)
            if submenu is None:
                return fail("order-submenu", shot2)
            self._save_rpa_shot(ts, "cancel_2_submenu", shot2, submenu)

            # 3) 'Cancel Order' is the only C item - typing it activates it
            pyautogui.press('c')
            time.sleep(0.4)

        self.log_event("CANCELLED resting order via tag menu")
        return True

    def manage_tracked_order(self, right_img):
        """Keep the resting limit attached to the midband line: cancel and
        re-place whenever the line moves. Stops once price touches the
        order level (assumed filled) or the tag disappears."""
        if not TRACK_ORDER or not self.tracked or self.paused:
            return
        if time.time() - self.tracked["time"] < TRACK_DELAY_AFTER_PLACE:
            return

        side = self.tracked["side"]
        line_bgr = BUY_LINE_BGR if side == "BUY" else SELL_LINE_BGR
        line_hit = detect_line(right_img, line_bgr)
        y_hint = line_hit[0] if line_hit is not None else self.tracked["y"]

        tag = find_order_tag(right_img, y_hint)
        if tag is None:
            # Tag gone: filled, manually cancelled, or briefly obscured
            self.tracked["tag_missing"] += 1
            if self.tracked["tag_missing"] >= TAG_MISSING_FRAMES:
                print(f"{side} order tag gone - assuming filled/closed, "
                      "tracking stopped")
                self.log_event(f"TRACKING STOPPED {side} - tag disappeared")
                self.tracked = None
            return
        self.tracked["tag_missing"] = 0

        if price_touched_level(right_img, tag[0]):
            print(f"{side} limit level touched by price - assuming FILLED, "
                  "tracking stopped")
            self.log_event(f"FILL ASSUMED {side} at y={tag[0]} - tracking stopped")
            self.tracked = None
            return

        if (line_hit is not None and
                abs(line_hit[0] - tag[0]) >= REPOSITION_THRESHOLD_PX and
                time.time() - self.last_reposition_time >= REPOSITION_MIN_INTERVAL):
            if self.cancel_order_via_menu(tag):
                time.sleep(0.3)
                self.click_limit_at(line_hit)
                self.tracked["y"] = line_hit[0]
                print(f"{side} limit MOVED with midband line "
                      f"(y {tag[0]} -> {line_hit[0]})")
                self.log_event(f"REPOSITION {side} y {tag[0]} -> {line_hit[0]}")
            self.last_reposition_time = time.time()

    def _save_rpa_shot(self, ts, name, img, box=None):
        vis = img.copy()
        if box is not None:
            cv2.rectangle(vis, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 3)
        cv2.imwrite(os.path.join("rpa_debug", f"{ts}_{name}.png"), vis)

    def run(self):
        if os.name == "posix":
            import sys, select, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def get_key():
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                if dr:
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

        print("Locating Close button...")
        self.find_button("CLOSE")
        if not os.path.exists(CLOSE_BUTTON_IMG):
            print("NOTE: close.png not found - using the DOM ratio position for "
                  "Close, verified by pixel color before every click (skipped "
                  "automatically if the DOM is not on screen).")

        print("Keys (console window must be focused): "
              "p=pause/resume  b=buy only  s=sell only  a=both  d=debug dump  q=quit")

        start_run = time.time()

        with mss.mss() as sct:
            mon = sct.monitors[1]
            mw, mh = mon["width"], mon["height"]

            gx1 = int(mw * LEFT_CHART[0])
            gy1 = int(mh * min(LEFT_CHART[1], RIGHT_CHART[1]))
            gx2 = int(mw * RIGHT_CHART[2])
            gy2 = int(mh * max(LEFT_CHART[3], RIGHT_CHART[3]))
            grab_region = {"left": mon["left"] + gx1, "top": mon["top"] + gy1,
                           "width": gx2 - gx1, "height": gy2 - gy1, "mon": 1}

            lx1 = int(mw * LEFT_CHART[0]) - gx1
            lx2 = int(mw * LEFT_CHART[2]) - gx1
            rx1 = int(mw * RIGHT_CHART[0]) - gx1
            rx2 = int(mw * RIGHT_CHART[2]) - gx1

            # Screen origins for translating 1510-crop pixels to click coords
            self.right_origin_x = grab_region["left"] + rx1
            self.chart_origin_y = grab_region["top"]

            try:
                while self.running:
                    full = np.array(sct.grab(grab_region))[:, :, :3]
                    left_img = np.ascontiguousarray(full[:, lx1:lx2, :])
                    right_img = np.ascontiguousarray(full[:, rx1:rx2, :])

                    if self.frame_count == 0:
                        cv2.imwrite("midband_left_debug.png", left_img)
                        cv2.imwrite("midband_right_debug.png", right_img)
                        print("Saved midband_left_debug.png (30020) and "
                              "midband_right_debug.png (15010) - verify!")

                    results = self.model.predict(
                        source=[left_img, right_img], verbose=False, imgsz=640,
                        device=device, half=self.use_half, conf=YOLO_CONF, iou=0.15)

                    left_dets = extract_detections(results[0])
                    right_dets = extract_detections(results[1])

                    # === EXIT: 1510 background area color transition ===
                    raw_area = classify_area(right_img[:, -AREA_EDGE_WIDTH:, :])
                    if raw_area == self.area_candidate:
                        self.area_frames += 1
                    else:
                        self.area_candidate = raw_area
                        self.area_frames = 1 if raw_area else 0

                    if (self.area_candidate is not None and
                            self.area_frames >= AREA_STABLE_FRAMES and
                            self.area_candidate != self.area_confirmed):
                        previous_area = self.area_confirmed
                        self.area_confirmed = self.area_candidate
                        print(f"1510 AREA: {self.area_confirmed} (was {previous_area or 'unknown'})")
                        if previous_area is not None and not self.paused:
                            if self.position == "LONG" and self.area_confirmed == "RED":
                                self.close_position("1510 area turned RED")
                            elif self.position == "SHORT" and self.area_confirmed == "GREEN":
                                self.close_position("1510 area turned GREEN")

                    # Keep the resting limit order attached to the midband
                    # line (cancel + re-place when the line moves)
                    self.manage_tracked_order(right_img)

                    signal, debug, matched_box = evaluate_frame(left_dets, right_dets)

                    # Re-arm the dedup whenever the 1510 RML changes, so
                    # EVERY new HH/LL can trade (even at a price level that
                    # already traded once)
                    r_lbl_now, _ = rightmost_label(*right_dets)
                    if r_lbl_now != self.prev_r_lbl:
                        self.prev_r_lbl = r_lbl_now
                        self.last_executed_sig = None

                    if signal is not None and signal == self.candidate_signal:
                        self.candidate_frames += 1
                    else:
                        self.candidate_signal = signal
                        self.candidate_frames = 1 if signal else 0

                    decision = None
                    if signal and self.candidate_frames >= STABLE_FRAMES and not self.paused:
                        now = time.time()
                        sig = (signal, round(matched_box[1] / Y_BUCKET) * Y_BUCKET)
                        cooldown_ok = (
                            (signal == "BUY" and self.mode in ("buy", "both")
                             and now - self.last_buy_time >= BUY_COOLDOWN) or
                            (signal == "SELL" and self.mode in ("sell", "both")
                             and now - self.last_sell_time >= SELL_COOLDOWN)
                        )
                        # Same-side adds are allowed: every NEW 1510 HH/LL
                        # places another entry; the area-color Close flattens
                        # everything (and cancels resting orders) on a flip
                        if cooldown_ok and sig != self.last_executed_sig:
                            decision, click_yx = self.place_limit_at_line(signal, right_img)
                            if decision:
                                self.last_executed_sig = sig
                                if signal == "BUY":
                                    self.last_buy_time = now
                                else:
                                    self.last_sell_time = now
                                self.log_event(f"LIMIT {decision} sig={sig} [{debug}]")
                                save_trade_snapshot(decision, left_img, right_img,
                                                    left_dets, right_dets, click_yx)

                    self.frame_count += 1
                    if decision or self.frame_count % STATUS_EVERY_N_FRAMES == 0:
                        fps = self.frame_count / max(0.001, time.time() - start_run)
                        print(f"Frame {self.frame_count}: {debug} signal={signal or '-'} "
                              f"area={self.area_confirmed or '-'} pos={self.position or '-'} "
                              f"mode={self.mode}{' PAUSED' if self.paused else ''} "
                              f"buys={self.buy_count} sells={self.sell_count} "
                              f"closes={self.close_count} ({fps:.1f} fps)")

                    key = get_key()
                    if key == 'p':
                        self.paused = not self.paused
                        print("PAUSED - no orders will be placed (press p to resume)"
                              if self.paused else "RESUMED - trading enabled")
                    elif key == 'b':
                        self.mode = "buy"
                        print("MODE: BUY ONLY - sell signals will be ignored")
                    elif key == 's':
                        self.mode = "sell"
                        print("MODE: SELL ONLY - buy signals will be ignored")
                    elif key == 'a':
                        self.mode = "both"
                        print("MODE: BOTH directions")
                    elif key == 'd':
                        cv2.imwrite("midband_left_debug.png", left_img)
                        cv2.imwrite("midband_right_debug.png", right_img)
                        for name, bgr in (("buy(green)", BUY_LINE_BGR),
                                          ("sell(magenta)", SELL_LINE_BGR)):
                            hit = detect_line(right_img, bgr)
                            print(f"  {name} line: "
                                  f"{'found at y=%d x=%d' % hit if hit else 'NOT FOUND'}")
                        print("Dumped midband_left_debug.png / midband_right_debug.png")
                    elif key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        runtime = time.time() - start_run
                        minutes, seconds = divmod(runtime, 60)
                        log_content = (
                            f"\n[midband] Time: {datetime.now().strftime('%H:%M')}  "
                            f"Date: {date.today()}\n"
                            f"Runtime: {int(minutes)} min {seconds:.2f} sec\n"
                            f"Frames: {self.frame_count} "
                            f"(avg {runtime / max(1, self.frame_count) * 1000:.0f} ms/frame)\n"
                            f"Final mode: {self.mode}\n"
                            f"Final number of buys: {self.buy_count}\n"
                            f"Final number of sells: {self.sell_count}\n"
                            f"Number of closes: {self.close_count}\n"
                            f"Open position at exit: {self.position or 'none'}\n"
                        )
                        print(log_content)
                        with open("log.txt", "a") as log_file:
                            log_file.write(log_content)
                        break

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting...")
            finally:
                self.finished.emit()


class MarketWorker:
    def __init__(self, trade_mode):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.detection_thread = MidbandWorker(trade_mode)
        self.detection_thread.finished.connect(self.on_finished)
        self.detection_thread.start()

    def on_finished(self):
        print("Detection finished.")
        self.app.quit()


if __name__ == "__main__":
    mode = input("Enter mode (buy / sell / both): ").strip().lower()
    while mode not in ("buy", "sell", "both"):
        mode = input("Invalid input, enter buy, sell, or both: ").strip().lower()

    mw = MarketWorker(mode)
    sys.exit(mw.app.exec_())
