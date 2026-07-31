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
# market_worker7.py
# YOLO label + black-candle strategy. Layout: 30020 chart LEFT, 15010 RIGHT.
#   BUY  (click Buy Mkt):  3020 RML == HH  AND  1510 RML == HL with a
#                          black candle formed ABOVE the HL label
#   SELL (click Sell Mkt): 3020 RML == LL  AND  1510 RML == LH with a
#                          black candle formed BELOW the LH label
# Built for speed: one screen grab per frame, both crops batched into a
# single GPU inference (FP16 on CUDA), buttons pre-located at startup.
# =====================================================================

pyautogui.PAUSE = 0.01

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

if platform.system() == "Darwin":
    MODEL_PATH = '/Users/ryanabbas/Desktop/work/StockMarket/yolo_models/candles_labels/weights/best.pt'
else:
    MODEL_PATH = 'c:/Users/ArshadParveez/Documents/Trading Code/StockMarket/yolo_models/candles_labels/weights/best.pt'

# Chart plot areas as ratios of the FULL screen (x1, y1, x2, y2).
# 30020 is the LEFT chart in this layout, 15010 the RIGHT chart.
LEFT_CHART = (0.00, 0.03, 0.42, 0.90)    # 30020
RIGHT_CHART = (0.465, 0.03, 0.83, 0.90)  # 15010

# DOM buttons (same scheme as market_worker6: template image first, ratio fallback)
BUY_BUTTON_RATIO = (0.905, 0.049)
SELL_BUTTON_RATIO = (0.967, 0.049)
CLOSE_BUTTON_RATIO = (0.967, 0.174)
BUY_BUTTON_IMG = "buy_mkt.png"
SELL_BUTTON_IMG = "sell_mkt.png"
CLOSE_BUTTON_IMG = "close.png"

# Exit rule: after a trade, watch the 1510 chart's background area color at
# its right edge (same signal as market_worker6). LONG closes when the area
# turns RED; SHORT closes when it turns GREEN. Transition-based: the color
# must CHANGE to the adverse color after entry.
AREA_EDGE_WIDTH = 150      # rightmost pixels of the 1510 crop to scan
AREA_STRIP_WIDTH = 25      # columns sampled at the found edge
AREA_MIN_PIXELS = 150
AREA_DOMINANCE = 1.5
AREA_STABLE_FRAMES = 3     # consecutive identical readings to confirm a color

LABEL_CONF = 0.30        # min confidence for HH/HL/LL/LH labels
CANDLE_CONF = 0.45       # min confidence for candle detections
YOLO_CONF = 0.25         # model-level confidence floor
STABLE_FRAMES = 2        # consecutive frames a signal must repeat before firing
BUY_COOLDOWN = 3.0       # seconds between buys
SELL_COOLDOWN = 3.0      # seconds between sells
X_TOLERANCE = 15         # candle center may be this far outside the label box (px)
EDGE_ZONE_RATIO = 0.30   # 1510 pattern label must be in the rightmost 30% of the
                         # chart - old labels that scrolled left are not tradeable
RML_STABLE_FRAMES = 3    # frames a 1510 label must hold to confirm an episode
MIN_SIGNAL_HOLD = 2.0    # seconds a signal must hold before it may trade -
                         # labels at the developing extreme repaint, and the
                         # old y-bucket dedup re-fired the SAME pattern as
                         # autoscaling drifted its pixel position (3 sells in
                         # 10s at y=1185/1170/1155 in the 19:08 run)
STATUS_EVERY_N_FRAMES = 30

VALID_LABELS = {"HH", "HL", "LL", "LH"}


def extract_detections(result):
    """Pull aligned (boxes, labels, scores) numpy lists from a YOLO result."""
    if result.boxes is None or len(result.boxes) == 0:
        return [], [], []
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    labels = [result.names[c] for c in cls]
    return list(xyxy), labels, list(conf)


def rightmost_label(boxes, labels, scores, min_conf=LABEL_CONF):
    """Rightmost HH/HL/LL/LH detection: (label, box) or (None, None)."""
    best = None
    for b, l, s in zip(boxes, labels, scores):
        if l in VALID_LABELS and s >= min_conf:
            if best is None or b[0] > best[1][0]:
                best = (l, b)
    return best if best else (None, None)


def classify_area(crop_bgr):
    """'GREEN'/'RED'/None for the background shading band at the right edge
    of a chart crop (ported from market_worker6). Pale shading only: bricks,
    lines and gray margins are excluded by saturation/value bounds."""
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


def top_labels(boxes, labels, scores, n=2, min_conf=LABEL_CONF):
    """Up to n rightmost HH/HL/LL/LH detections as [(label, box), ...]."""
    entries = [
        (l, b) for b, l, s in zip(boxes, labels, scores)
        if l in VALID_LABELS and s >= min_conf
    ]
    entries.sort(key=lambda e: e[1][0], reverse=True)
    return entries[:n]


def all_candles(boxes, labels, scores, min_conf=CANDLE_CONF):
    """All candle detection boxes above the confidence floor."""
    return [b for b, l, s in zip(boxes, labels, scores)
            if l == "candle" and s >= min_conf]


def candle_matches_label(candle, label_box, side):
    """True if the candle sits at the label's x-position and on the correct
    side vertically: above the label for BUY (HL), below for SELL (LH)."""
    if candle is None or label_box is None:
        return False
    cx = (candle[0] + candle[2]) // 2
    if not (label_box[0] - X_TOLERANCE <= cx <= label_box[2] + X_TOLERANCE):
        return False
    candle_cy = (candle[1] + candle[3]) // 2
    label_cy = (label_box[1] + label_box[3]) // 2
    return candle_cy < label_cy if side == "BUY" else candle_cy > label_cy


def find_matching_candle(candles, label_box, side):
    """First candle (any candle, not just the rightmost) matching the label."""
    for c in candles:
        if candle_matches_label(c, label_box, side):
            return c
    return None


def evaluate_frame(left_dets, right_dets, right_width=None):
    """Decide this frame. Returns (signal, debug, matched_label_box, near_miss):
    signal 'BUY'/'SELL'/None; near_miss describes a partial pattern for the
    diagnostic log when no trade fires. If right_width is given, the pattern
    label must sit in the rightmost EDGE_ZONE_RATIO of the 1510 chart."""
    l_lbl, l_box = rightmost_label(*left_dets)
    tops = top_labels(*right_dets)
    candles = all_candles(*right_dets)

    top_names = "/".join(t[0] for t in tops) or "-"
    debug = f"3020={l_lbl or '-'} 1510={top_names} candles={len(candles)}"

    edge_min_x = right_width * (1 - EDGE_ZONE_RATIO) if right_width else None

    near_miss = None
    if l_lbl == "HH":
        for r_lbl, r_box in tops:
            if r_lbl == "HL":
                if edge_min_x is not None and r_box[0] < edge_min_x:
                    near_miss = f"HH+HL but HL too far left (stale) at x={r_box[0]} [{debug}]"
                    continue
                if find_matching_candle(candles, r_box, "BUY") is not None:
                    return "BUY", debug, r_box, None
                near_miss = f"HH+HL but no candle above HL at x={r_box[0]} [{debug}]"
    elif l_lbl == "LL":
        for r_lbl, r_box in tops:
            if r_lbl == "LH":
                if edge_min_x is not None and r_box[0] < edge_min_x:
                    near_miss = f"LL+LH but LH too far left (stale) at x={r_box[0]} [{debug}]"
                    continue
                if find_matching_candle(candles, r_box, "SELL") is not None:
                    return "SELL", debug, r_box, None
                near_miss = f"LL+LH but no candle below LH at x={r_box[0]} [{debug}]"

    return None, debug, None, near_miss


def save_trade_snapshot(tag, left_img, right_img, left_dets, right_dets):
    """Save annotated copies of both crops showing every detection, so any
    questionable trade can be reviewed after the fact."""
    os.makedirs("trade_snaps", exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    for name, img, dets in (("left", left_img, left_dets),
                            ("right", right_img, right_dets)):
        vis = img.copy()
        for b, l, s in zip(*dets):
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(vis, f"{l} {s:.2f}", (b[0], max(12, b[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join("trade_snaps", f"{ts}_{tag}_{name}.png"), vis)


class Worker7(QThread):
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
        self.candidate_signal = None
        self.candidate_frames = 0
        self.candidate_since = 0.0
        self.button_pos = {}

        # One trade per label EPISODE (ported from Midband_entry): a BUY on
        # an HL disarms buys until a DIFFERENT confirmed 1510 label appears;
        # mirror for SELL/LH. Replaces the y-bucket dedup, which re-fired
        # the same pattern as chart autoscaling drifted label pixels.
        self.entry_armed = {"BUY": True, "SELL": True}
        self.rml_confirmed = None
        self.rml_candidate = None
        self.rml_candidate_frames = 0

        # Position + 1510 area-color exit tracking
        self.position = None          # None | "LONG" | "SHORT"
        self.close_count = 0
        self.area_confirmed = None    # 'GREEN'/'RED' after AREA_STABLE_FRAMES
        self.area_candidate = None
        self.area_frames = 0

        print("Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)
        if device != 'cpu':
            self.model.to(device)
        self.use_half = (device == 'cuda')
        # Warmup so the first real frame is not slow
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(source=[dummy, dummy], verbose=False, imgsz=640,
                           device=device, half=self.use_half, conf=YOLO_CONF, iou=0.15)
        print("Model ready.")

    def find_button(self, which):
        if which in self.button_pos:
            return self.button_pos[which]
        img = {"BUY": BUY_BUTTON_IMG, "SELL": SELL_BUTTON_IMG,
               "CLOSE": CLOSE_BUTTON_IMG}[which]
        pos = None
        if os.path.exists(img):
            try:
                loc = pyautogui.locateCenterOnScreen(img, confidence=0.85)
                if loc:
                    pos = (int(loc.x), int(loc.y))
                    print(f"{which} button located by image at {pos}")
            except Exception as e:
                print(f"Template search for {which} failed ({e}), using ratio fallback")
        if pos is None:
            sw, sh = pyautogui.size()
            ratio = {"BUY": BUY_BUTTON_RATIO, "SELL": SELL_BUTTON_RATIO,
                     "CLOSE": CLOSE_BUTTON_RATIO}[which]
            pos = (int(sw * ratio[0]), int(sh * ratio[1]))
            print(f"{which} button using ratio position {pos}")
        self.button_pos[which] = pos
        return pos

    def execute_trade(self, side):
        x, y = self.find_button(side)
        pyautogui.click(x, y)
        if side == "BUY":
            self.buy_count += 1
            self.position = "LONG"
        else:
            self.sell_count += 1
            self.position = "SHORT"
        return side

    def close_position(self, reason):
        x, y = self.find_button("CLOSE")
        pyautogui.click(x, y)
        self.close_count += 1
        print(f"CLOSED {self.position} - {reason}")
        with open("worker7_debug.log", "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} "
                    f"CLOSE {self.position} - {reason}\n")
        self.position = None

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

        print("Locating DOM buttons...")
        for which in ("BUY", "SELL", "CLOSE"):
            self.find_button(which)

        print("Keys (console window must be focused): "
              "p=pause/resume  b=buy only  s=sell only  a=both  d=debug dump  q=quit")

        start_run = time.time()
        last_near_miss_log = 0.0

        with mss.mss() as sct:
            mon = sct.monitors[1]
            mw, mh = mon["width"], mon["height"]

            # One grab covering both charts, then slice
            gx1 = int(mw * LEFT_CHART[0])
            gy1 = int(mh * min(LEFT_CHART[1], RIGHT_CHART[1]))
            gx2 = int(mw * RIGHT_CHART[2])
            gy2 = int(mh * max(LEFT_CHART[3], RIGHT_CHART[3]))
            grab_region = {"left": mon["left"] + gx1, "top": mon["top"] + gy1,
                           "width": gx2 - gx1, "height": gy2 - gy1, "mon": 1}

            # Slice offsets within the grabbed image
            lx1 = int(mw * LEFT_CHART[0]) - gx1
            lx2 = int(mw * LEFT_CHART[2]) - gx1
            rx1 = int(mw * RIGHT_CHART[0]) - gx1
            rx2 = int(mw * RIGHT_CHART[2]) - gx1

            try:
                while self.running:
                    full = np.array(sct.grab(grab_region))[:, :, :3]
                    left_img = np.ascontiguousarray(full[:, lx1:lx2, :])
                    right_img = np.ascontiguousarray(full[:, rx1:rx2, :])

                    if self.frame_count == 0:
                        cv2.imwrite("worker7_left_debug.png", left_img)
                        cv2.imwrite("worker7_right_debug.png", right_img)
                        print("Saved worker7_left_debug.png (should show 30020) and "
                              "worker7_right_debug.png (should show 15010) - verify!")

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
                        # Close only on a transition to the adverse color
                        if previous_area is not None and not self.paused:
                            if self.position == "LONG" and self.area_confirmed == "RED":
                                self.close_position("1510 area turned RED")
                            elif self.position == "SHORT" and self.area_confirmed == "GREEN":
                                self.close_position("1510 area turned GREEN")

                    signal, debug, matched_box, near_miss = evaluate_frame(
                        left_dets, right_dets, right_img.shape[1])

                    # Diagnostic: record partial patterns so missed trades are explainable
                    if near_miss and time.time() - last_near_miss_log >= 1.0:
                        last_near_miss_log = time.time()
                        with open("worker7_debug.log", "a") as f:
                            f.write(f"{datetime.now().strftime('%H:%M:%S')} NEAR-MISS: {near_miss}\n")

                    # Label-episode tracking: confirm the 1510 RML over
                    # RML_STABLE_FRAMES frames; a confirmed DIFFERENT label
                    # re-arms the entries it separates (non-HL re-arms BUY,
                    # non-LH re-arms SELL)
                    r_lbl_now, _ = rightmost_label(*right_dets)
                    if r_lbl_now == self.rml_candidate:
                        self.rml_candidate_frames += 1
                    else:
                        self.rml_candidate = r_lbl_now
                        self.rml_candidate_frames = 1
                    if (self.rml_candidate is not None and
                            self.rml_candidate_frames >= RML_STABLE_FRAMES and
                            self.rml_candidate != self.rml_confirmed):
                        self.rml_confirmed = self.rml_candidate
                        if self.rml_confirmed != "HL" and not self.entry_armed["BUY"]:
                            self.entry_armed["BUY"] = True
                            print(f"1510 RML now {self.rml_confirmed} - "
                                  "BUY re-armed for the next HL pattern")
                        if self.rml_confirmed != "LH" and not self.entry_armed["SELL"]:
                            self.entry_armed["SELL"] = True
                            print(f"1510 RML now {self.rml_confirmed} - "
                                  "SELL re-armed for the next LH pattern")

                    # Debounce: same signal on STABLE_FRAMES consecutive
                    # frames AND held for MIN_SIGNAL_HOLD seconds (labels on
                    # the developing brick are provisional and repaint)
                    if signal is not None and signal == self.candidate_signal:
                        self.candidate_frames += 1
                    else:
                        self.candidate_signal = signal
                        self.candidate_frames = 1 if signal else 0
                        self.candidate_since = time.time()

                    decision = None
                    if (signal and self.candidate_frames >= STABLE_FRAMES and
                            time.time() - self.candidate_since >= MIN_SIGNAL_HOLD and
                            not self.paused):
                        now = time.time()
                        cooldown_ok = (
                            (signal == "BUY" and self.mode in ("buy", "both")
                             and now - self.last_buy_time >= BUY_COOLDOWN) or
                            (signal == "SELL" and self.mode in ("sell", "both")
                             and now - self.last_sell_time >= SELL_COOLDOWN)
                        )
                        if cooldown_ok and self.entry_armed[signal]:
                            decision = self.execute_trade(signal)
                            self.entry_armed[signal] = False
                            if signal == "BUY":
                                self.last_buy_time = now
                            else:
                                self.last_sell_time = now
                            print(f"TRADE: {decision} [{debug}]")
                            # Forensics: log + annotated snapshots of what was seen
                            with open("worker7_debug.log", "a") as f:
                                f.write(f"{datetime.now().strftime('%H:%M:%S')} "
                                        f"TRADE {decision} [{debug}]\n")
                            save_trade_snapshot(decision, left_img, right_img,
                                                left_dets, right_dets)

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
                        print("PAUSED - no trades will be placed (press p to resume)"
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
                        cv2.imwrite("worker7_left_debug.png", left_img)
                        cv2.imwrite("worker7_right_debug.png", right_img)
                        print("Dumped worker7_left_debug.png / worker7_right_debug.png")
                    elif key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        runtime = time.time() - start_run
                        minutes, seconds = divmod(runtime, 60)
                        log_content = (
                            f"\n[worker7] Time: {datetime.now().strftime('%H:%M')}  "
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
        self.detection_thread = Worker7(trade_mode)
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
