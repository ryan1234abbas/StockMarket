import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
import mss
import cv2
import os
import pyautogui
import platform
from datetime import date, datetime

# =====================================================================
# Confluence_entry.py
# Full-alignment strategy on the RIGHTMOST chart (MES 1510 in the
# user's 4-chart layout). All conditions must agree:
#   BUY : background GREEN + Midband GREEN + Large Triggers GREEN +
#         Small Triggers GREEN + BBs ABOVE the BB Zero Line +
#         RSI Lines ABOVE the dotted 50 line          -> click Buy Mkt
#   SELL: background MAGENTA + Midband MAGENTA + Large Triggers MAGENTA
#         + Small Triggers MAGENTA + BBs BELOW zero +
#         RSI Lines BELOW the dotted 50 line          -> click Sell Mkt
# No YOLO - pure color/position detection, ~20+ checks/sec.
# Fires ONCE per alignment episode: after a trade the side re-arms only
# after the alignment breaks. Signals must hold MIN_SIGNAL_HOLD seconds.
# Press 'd' for a calibration dump (confluence_debug.png) showing every
# detector's verdict - use it to tune the color constants below.
# =====================================================================

pyautogui.PAUSE = 0.01

# Panels of the RIGHTMOST chart as ratios of the FULL screen (x1,y1,x2,y2),
# measured from the user's reference screenshot (4-chart layout, DOM far right)
PRICE_PANEL = (0.700, 0.030, 0.863, 0.715)
RSI_PANEL   = (0.700, 0.722, 0.863, 0.782)
BB_PANEL    = (0.700, 0.790, 0.863, 0.943)
# The panel boxes intentionally overshoot into the price axis; the true
# plot edge is FOUND from the background shading each frame, and only a
# window hugging that edge is analyzed - so color flips register
# immediately instead of waiting to fill a stale wide window.
# Price uses a TIGHT window for fast trigger-color response. RSI/BB need
# more horizontal width to separate the moving lines from the fixed
# 50/zero reference lines (in a very narrow window a moving line looks
# horizontal and gets mistaken for a reference line).
TIGHT_WINDOW = 40
RSIBB_WINDOW = 120

# DOM buttons (template image first, ratio fallback measured from the layout)
BUY_BUTTON_RATIO = (0.9275, 0.0536)
SELL_BUTTON_RATIO = (0.9740, 0.0536)
BUY_BUTTON_IMG = "buy_mkt.png"
SELL_BUTTON_IMG = "sell_mkt.png"

# --- color definitions (BGR centers +/- tolerance). Tune via 'd' dumps. ---
MIDBAND_GREEN_BGR = (4, 255, 129)     # thick chartreuse band (as Midband_entry)
MIDBAND_MAGENTA_BGR = (255, 4, 255)   # thick magenta band
MIDBAND_TOL = 60
MIDBAND_MIN_BLOB = 80    # px; the midband is thick - small flecks don't count

LARGE_GREEN_HSV_LO = (35, 100, 50)    # dark-green thick trigger curves
LARGE_GREEN_HSV_HI = (85, 255, 190)
LARGE_MIN_PIXELS = 40

SMALL_GREEN_HSV_LO = (35, 120, 170)   # bright thin green trigger lines
SMALL_GREEN_HSV_HI = (85, 255, 255)
SMALL_RED_HSV_LO1 = (0, 140, 140)     # bright thin red trigger lines
SMALL_RED_HSV_HI1 = (10, 255, 255)
SMALL_RED_HSV_LO2 = (170, 140, 140)
SMALL_RED_HSV_HI2 = (180, 255, 255)
SMALL_MIN_PIXELS = 20
LINE_MAX_FILL = 0.45     # line-like components only (bricks are solid boxes)
LINE_MIN_SPAN = 20       # a line component spans at least this many px
                         # (the analysis window is only TIGHT_WINDOW wide)

BG_GREEN_HUE = (30, 90)      # pale background shading, low saturation
BG_MAGENTA_HUE = (110, 175)
BG_SAT = (8, 110)
BG_VAL_MIN = 100
BG_MIN_PIXELS = 150

RSI_BLUE = dict(b_min=170, g_max=140, r_max=140)   # RSI line colors
RSI_RED = dict(r_min=170, g_max=140, b_max=140)
DOTTED_DARK_MAX = 110    # 50-line dots are dark
HLINE_ROW_FRACTION = 0.55  # a row this full of one color = horizontal marker,
                           # excluded when finding the moving RSI lines

BB_ZERO_BLUE = dict(b_min=170, g_max=150, r_max=120)  # solid blue zero line
BB_DOT_GREEN_HSV = ((35, 90, 90), (85, 255, 255))     # dotted BB curves
BB_DOT_RED_HSV1 = ((0, 90, 90), (10, 255, 255))
BB_DOT_RED_HSV2 = ((170, 90, 90), (180, 255, 255))
BB_MIN_PIXELS = 12

# --- pacing (protections proven in the other bots) ---
STABLE_FRAMES = 3        # consecutive frames of full alignment
MIN_SIGNAL_HOLD = 2.0    # seconds the alignment must hold before trading
BUY_COOLDOWN = 3.0
SELL_COOLDOWN = 3.0
STATUS_EVERY_N_FRAMES = 30


def _bg_mask(win_bgr):
    """Mask of the pale background shading (green or magenta family)."""
    hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    base = (s >= BG_SAT[0]) & (s <= BG_SAT[1]) & (v >= BG_VAL_MIN)
    green = base & (h >= BG_GREEN_HUE[0]) & (h <= BG_GREEN_HUE[1])
    magenta = base & (h >= BG_MAGENTA_HUE[0]) & (h <= BG_MAGENTA_HUE[1])
    return (green | magenta)


def plot_right_edge_col(price_bgr):
    """Rightmost column of the chart PLOT area, found via the background
    shading - everything right of it is the price axis (gray, no shading).
    All panels share the same x range, so one edge serves all three."""
    mask = _bg_mask(price_bgr)
    h = price_bgr.shape[0]
    cols = mask.sum(axis=0)
    shaded = np.where(cols >= h * 0.25)[0]
    return int(shaded[-1]) + 1 if shaded.size else price_bgr.shape[1]


def _mask_components(mask, min_span=0, max_fill=1.0, min_area=1):
    """Filter a binary mask to components that look line-like."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    total = 0
    for i in range(1, n):
        x, y, w, h, area = [int(stats[i, k]) for k in range(5)]
        span = max(w, h)
        fill = area / max(1, w * h)
        if area >= min_area and span >= min_span and fill <= max_fill:
            keep[lbl == i] = 255
            total += area
    return keep, total


def detect_background(win_bgr):
    """'GREEN' / 'MAGENTA' / None from the pale background shading."""
    hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    base = (s >= BG_SAT[0]) & (s <= BG_SAT[1]) & (v >= BG_VAL_MIN)
    green = int((base & (h >= BG_GREEN_HUE[0]) & (h <= BG_GREEN_HUE[1])).sum())
    magenta = int((base & (h >= BG_MAGENTA_HUE[0]) & (h <= BG_MAGENTA_HUE[1])).sum())
    if green >= BG_MIN_PIXELS and green > magenta * 1.5:
        return "GREEN"
    if magenta >= BG_MIN_PIXELS and magenta > green * 1.5:
        return "MAGENTA"
    return None


def detect_midband(win_bgr):
    """'GREEN' / 'MAGENTA' / None from the thick midband. Uses the largest
    connected blob so thin dashed lines of similar colors don't count."""
    def blob(center):
        t = np.array(center, np.int16)
        lo = np.clip(t - MIDBAND_TOL, 0, 255).astype(np.uint8)
        hi = np.clip(t + MIDBAND_TOL, 0, 255).astype(np.uint8)
        mask = cv2.inRange(win_bgr, lo, hi)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        return max((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)),
                   default=0)
    g = blob(MIDBAND_GREEN_BGR)
    m = blob(MIDBAND_MAGENTA_BGR)
    if g >= MIDBAND_MIN_BLOB and g >= m:
        return "GREEN"
    if m >= MIDBAND_MIN_BLOB:
        return "MAGENTA"
    return None


def detect_large_triggers(win_bgr):
    """'GREEN' / 'MAGENTA' by DOMINANCE in the tight edge window - no
    green-first bias, so a color flip registers as soon as the new color
    outweighs the old one at the right edge."""
    hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
    green = int(cv2.inRange(hsv, LARGE_GREEN_HSV_LO, LARGE_GREEN_HSV_HI).sum() // 255)
    t = np.array(MIDBAND_MAGENTA_BGR, np.int16)
    lo = np.clip(t - MIDBAND_TOL, 0, 255).astype(np.uint8)
    hi = np.clip(t + MIDBAND_TOL, 0, 255).astype(np.uint8)
    magenta = int(cv2.inRange(win_bgr, lo, hi).sum() // 255)
    if green >= LARGE_MIN_PIXELS and green > magenta * 1.2:
        return "GREEN"
    if magenta >= LARGE_MIN_PIXELS and magenta > green * 1.2:
        return "MAGENTA"
    return None


def detect_small_triggers(win_bgr):
    """'GREEN' / 'MAGENTA' from the thin bright trigger lines. Components
    must be LINE-like (long and sparse) so solid price bricks don't count."""
    hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
    gmask = cv2.inRange(hsv, SMALL_GREEN_HSV_LO, SMALL_GREEN_HSV_HI)
    rmask = cv2.bitwise_or(cv2.inRange(hsv, SMALL_RED_HSV_LO1, SMALL_RED_HSV_HI1),
                           cv2.inRange(hsv, SMALL_RED_HSV_LO2, SMALL_RED_HSV_HI2))
    _, g = _mask_components(gmask, LINE_MIN_SPAN, LINE_MAX_FILL, SMALL_MIN_PIXELS)
    _, r = _mask_components(rmask, LINE_MIN_SPAN, LINE_MAX_FILL, SMALL_MIN_PIXELS)
    if g >= SMALL_MIN_PIXELS and g > r:
        return "GREEN"
    if r >= SMALL_MIN_PIXELS:
        return "MAGENTA"
    return None


def _horizontal_rows(mask, min_fraction):
    """Row indices where the mask covers most of the width (horizontal
    marker lines like the dashed 70/30 or the solid zero line)."""
    w = mask.shape[1]
    counts = (mask > 0).sum(axis=1)
    return np.where(counts >= w * min_fraction)[0]


def detect_rsi(win_bgr):
    """'ABOVE' when the RSI lines sit above the dotted 50 line at the right
    edge, 'BELOW' when below, None when mixed/undetected."""
    b = win_bgr[:, :, 0].astype(np.int16)
    g = win_bgr[:, :, 1].astype(np.int16)
    r = win_bgr[:, :, 2].astype(np.int16)

    blue = ((b >= RSI_BLUE["b_min"]) & (g <= RSI_BLUE["g_max"]) &
            (r <= RSI_BLUE["r_max"])).astype(np.uint8) * 255
    red = ((r >= RSI_RED["r_min"]) & (g <= RSI_RED["g_max"]) &
           (b <= RSI_RED["b_max"])).astype(np.uint8) * 255

    # Drop horizontal marker rows (dashed levels) so only the moving lines remain
    for m in (blue, red):
        for row in _horizontal_rows(m, HLINE_ROW_FRACTION):
            m[row, :] = 0

    # The dotted 50 line: the row with the most dark dots
    dark = ((b <= DOTTED_DARK_MAX) & (g <= DOTTED_DARK_MAX) &
            (r <= DOTTED_DARK_MAX)).astype(np.uint8)
    row_counts = dark.sum(axis=1)
    if row_counts.max() < 4:
        return None, None
    y50 = int(np.argmax(row_counts))

    def line_y(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        cutoff = xs.max() - 25   # rightmost stretch of the line
        sel = xs >= cutoff
        return float(ys[sel].mean()) if sel.any() else None

    by, ry = line_y(blue), line_y(red)
    if by is None or ry is None:
        return None, y50
    if by < y50 and ry < y50:
        return "ABOVE", y50
    if by > y50 and ry > y50:
        return "BELOW", y50
    return None, y50


def detect_bb(win_bgr):
    """'ABOVE' when the dotted BB curves sit above the solid blue zero line
    at the right edge, 'BELOW' when below."""
    b = win_bgr[:, :, 0].astype(np.int16)
    g = win_bgr[:, :, 1].astype(np.int16)
    r = win_bgr[:, :, 2].astype(np.int16)
    blue = ((b >= BB_ZERO_BLUE["b_min"]) & (g <= BB_ZERO_BLUE["g_max"]) &
            (r <= BB_ZERO_BLUE["r_max"])).astype(np.uint8) * 255
    zero_rows = _horizontal_rows(blue, HLINE_ROW_FRACTION)
    if zero_rows.size == 0:
        return None, None
    y0 = int(zero_rows.mean())

    hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
    dots = cv2.inRange(hsv, *BB_DOT_GREEN_HSV)
    dots = cv2.bitwise_or(dots, cv2.inRange(hsv, *BB_DOT_RED_HSV1))
    dots = cv2.bitwise_or(dots, cv2.inRange(hsv, *BB_DOT_RED_HSV2))
    ys, xs = np.where(dots > 0)
    if len(xs) < BB_MIN_PIXELS:
        return None, y0
    cutoff = xs.max() - 25
    sel = xs >= cutoff
    if not sel.any():
        return None, y0
    yb = float(ys[sel].mean())
    return ("ABOVE" if yb < y0 else "BELOW"), y0


def diagnostics(price_win, rsi_win, bb_win):
    """Raw measurements behind each verdict, for the 'd' calibration dump
    and the log - so a blocking condition can be seen, not guessed."""
    hsv = cv2.cvtColor(price_win, cv2.COLOR_BGR2HSV)
    bgm = _bg_mask(price_win)
    h = hsv[:, :, 0]
    bg_g = int((bgm & (h >= BG_GREEN_HUE[0]) & (h <= BG_GREEN_HUE[1])).sum())
    bg_m = int((bgm & (h >= BG_MAGENTA_HUE[0]) & (h <= BG_MAGENTA_HUE[1])).sum())

    def blob(center):
        t = np.array(center, np.int16)
        lo = np.clip(t - MIDBAND_TOL, 0, 255).astype(np.uint8)
        hi = np.clip(t + MIDBAND_TOL, 0, 255).astype(np.uint8)
        m = cv2.inRange(price_win, lo, hi)
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        return max((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n)), default=0)

    lg = int(cv2.inRange(hsv, LARGE_GREEN_HSV_LO, LARGE_GREEN_HSV_HI).sum() // 255)
    _, rsi_y = detect_rsi(rsi_win)
    _, bb_y = detect_bb(bb_win)
    return (f"bg[g={bg_g},m={bg_m}] mid[g={blob(MIDBAND_GREEN_BGR)},"
            f"m={blob(MIDBAND_MAGENTA_BGR)}] large[g={lg}] "
            f"rsi[y50={rsi_y}] bb[y0={bb_y}]")


def evaluate(price_win, rsi_win, bb_win):
    """Returns (signal, states-dict). BUY only when every condition is
    green/above; SELL only when every condition is magenta/below."""
    states = {
        "bg": detect_background(price_win),
        "mid": detect_midband(price_win),
        "large": detect_large_triggers(price_win),
        "small": detect_small_triggers(price_win),
    }
    states["rsi"], _ = detect_rsi(rsi_win)
    states["bb"], _ = detect_bb(bb_win)

    if (states["bg"] == "GREEN" and states["mid"] == "GREEN" and
            states["large"] == "GREEN" and states["small"] == "GREEN" and
            states["rsi"] == "ABOVE" and states["bb"] == "ABOVE"):
        return "BUY", states
    if (states["bg"] == "MAGENTA" and states["mid"] == "MAGENTA" and
            states["large"] == "MAGENTA" and states["small"] == "MAGENTA" and
            states["rsi"] == "BELOW" and states["bb"] == "BELOW"):
        return "SELL", states
    return None, states


class ConfluenceWorker(QThread):
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
        self.button_pos = {}

        self.candidate_signal = None
        self.candidate_frames = 0
        self.candidate_since = 0.0
        # Fire once per alignment episode; re-arm when alignment breaks
        self.entry_armed = {"BUY": True, "SELL": True}

    def log_event(self, text):
        with open("confluence_debug.log", "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} {text}\n")

    def find_button(self, which):
        if which in self.button_pos:
            return self.button_pos[which]
        img = BUY_BUTTON_IMG if which == "BUY" else SELL_BUTTON_IMG
        pos = None
        if os.path.exists(img):
            try:
                loc = pyautogui.locateCenterOnScreen(img, confidence=0.85)
                if loc:
                    pos = (int(loc.x), int(loc.y))
                    print(f"{which} button located by image at {pos}")
            except Exception as e:
                print(f"Template search for {which} failed ({e}), using ratio")
        if pos is None:
            sw, sh = pyautogui.size()
            ratio = BUY_BUTTON_RATIO if which == "BUY" else SELL_BUTTON_RATIO
            pos = (int(sw * ratio[0]), int(sh * ratio[1]))
            print(f"{which} button using ratio position {pos}")
        self.button_pos[which] = pos
        return pos

    def execute_trade(self, side):
        x, y = self.find_button(side)
        pyautogui.click(x, y)
        if side == "BUY":
            self.buy_count += 1
        else:
            self.sell_count += 1
        return side

    def save_region_overlay(self):
        """Full-screen capture with the three panel boxes drawn on it, so the
        panel ratios can be checked against the real subpanels. This is the
        ground-truth artifact for placing PRICE/RSI/BB_PANEL correctly."""
        with mss.mss() as sct:
            mon = sct.monitors[1]
            full = np.array(sct.grab(mon))[:, :, :3].copy()
        mh, mw = full.shape[:2]
        for name, box, color in (("PRICE", PRICE_PANEL, (0, 0, 255)),
                                  ("RSI", RSI_PANEL, (0, 255, 0)),
                                  ("BB", BB_PANEL, (255, 0, 0))):
            x1, y1, x2, y2 = box
            p1 = (int(mw * x1), int(mh * y1))
            p2 = (int(mw * x2), int(mh * y2))
            cv2.rectangle(full, p1, p2, color, 3)
            cv2.putText(full, name, (p1[0] + 5, p1[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imwrite("confluence_regions.png", full)
        print("Saved confluence_regions.png (full screen with panel boxes) - "
              "check the RSI/BB boxes land on the real subpanels")

    def save_calibration(self, price, rsi, bb, states):
        # Upscale each panel to a readable fixed width and stack them, so the
        # verdicts and the actual chart geometry are legible in the dump.
        W = 340

        def up(img, tag):
            scaled = cv2.resize(img, (W, max(40, int(img.shape[0] * W / img.shape[1]))),
                                interpolation=cv2.INTER_NEAREST)
            cv2.putText(scaled, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
            cv2.putText(scaled, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            return scaled

        sep = np.full((3, W, 3), 200, np.uint8)
        header = np.full((70, W, 3), 30, np.uint8)
        want = "BUY: all GREEN/ABOVE   SELL: all MAGENTA/BELOW"
        line1 = f"bg={states['bg']} mid={states['mid']} large={states['large']}"
        line2 = f"small={states['small']} rsi={states['rsi']} bb={states['bb']}"
        for i, t in enumerate((line1, line2)):
            cv2.putText(header, t, (6, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1)
        vis = np.vstack([header,
                         up(price, "PRICE (bg/mid/large/small)"), sep,
                         up(rsi, "RSI panel"), sep,
                         up(bb, "BB panel")])
        cv2.imwrite("confluence_debug.png", vis)
        diag = diagnostics(price, rsi, bb)
        txt = "  ".join(f"{k}={v or '?'}" for k, v in states.items())
        print(f"Saved confluence_debug.png [{txt}]  {diag}")
        self.log_event(f"CALIB {txt}  {diag}")

    def run(self):
        if os.name == "posix":
            import sys, select, tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def get_key():
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                return sys.stdin.read(1).lower() if dr else None

            import atexit
            atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old))
        else:
            import msvcrt

            def get_key():
                if msvcrt.kbhit():
                    return msvcrt.getch().decode("utf-8").lower()
                return None

        print("Locating DOM buttons...")
        for which in ("BUY", "SELL"):
            self.find_button(which)
        print("Keys (console focused): p=pause  b=buy only  s=sell only  "
              "a=both  d=calibration dump  q=quit")

        start_run = time.time()

        with mss.mss() as sct:
            mon = sct.monitors[1]
            mw, mh = mon["width"], mon["height"]

            def panel_region(box):
                x1, y1, x2, y2 = box
                return {"left": mon["left"] + int(mw * x1),
                        "top": mon["top"] + int(mh * y1),
                        "width": int(mw * (x2 - x1)),
                        "height": int(mh * (y2 - y1)), "mon": 1}

            regions = {name: panel_region(box) for name, box in
                       (("price", PRICE_PANEL), ("rsi", RSI_PANEL),
                        ("bb", BB_PANEL))}

            try:
                while self.running:
                    fulls = {name: np.array(sct.grab(reg))[:, :, :3]
                             for name, reg in regions.items()}
                    # Locate the true plot edge from the price panel's
                    # background shading (all panels share the same x range),
                    # then analyze only the tight window hugging it
                    edge = plot_right_edge_col(fulls["price"])
                    xp = max(0, edge - TIGHT_WINDOW)
                    xw = max(0, edge - RSIBB_WINDOW)
                    grabs = {
                        "price": np.ascontiguousarray(fulls["price"][:, xp:edge, :]),
                        "rsi": np.ascontiguousarray(fulls["rsi"][:, xw:edge, :]),
                        "bb": np.ascontiguousarray(fulls["bb"][:, xw:edge, :]),
                    }

                    signal, states = evaluate(grabs["price"], grabs["rsi"],
                                              grabs["bb"])

                    if self.frame_count == 0:
                        self.save_calibration(grabs["price"], grabs["rsi"],
                                              grabs["bb"], states)
                        self.save_region_overlay()

                    # Re-arm a side once the alignment is broken
                    if signal != "BUY" and not self.entry_armed["BUY"]:
                        self.entry_armed["BUY"] = True
                        print("Alignment broken - BUY re-armed")
                    if signal != "SELL" and not self.entry_armed["SELL"]:
                        self.entry_armed["SELL"] = True
                        print("Alignment broken - SELL re-armed")

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
                            txt = "  ".join(f"{k}={v}" for k, v in states.items())
                            print(f"TRADE: {decision} [{txt}]")
                            self.log_event(f"TRADE {decision} [{txt}]")
                            self.save_calibration(grabs["price"], grabs["rsi"],
                                                  grabs["bb"], states)
                            os.makedirs("trade_snaps", exist_ok=True)
                            cv2.imwrite(os.path.join(
                                "trade_snaps",
                                f"{datetime.now().strftime('%H%M%S')}_{decision}_confluence.png"),
                                grabs["price"])

                    self.frame_count += 1
                    if decision or self.frame_count % STATUS_EVERY_N_FRAMES == 0:
                        fps = self.frame_count / max(0.001, time.time() - start_run)
                        txt = " ".join(f"{k}={v or '-'}" for k, v in states.items())
                        print(f"Frame {self.frame_count}: {txt} "
                              f"signal={signal or '-'} mode={self.mode}"
                              f"{' PAUSED' if self.paused else ''} "
                              f"buys={self.buy_count} sells={self.sell_count} "
                              f"({fps:.1f} fps)")

                    key = get_key()
                    if key == 'p':
                        self.paused = not self.paused
                        print("PAUSED" if self.paused else "RESUMED")
                    elif key == 'b':
                        self.mode = "buy"
                        print("MODE: BUY ONLY")
                    elif key == 's':
                        self.mode = "sell"
                        print("MODE: SELL ONLY")
                    elif key == 'a':
                        self.mode = "both"
                        print("MODE: BOTH")
                    elif key == 'd':
                        self.save_calibration(grabs["price"], grabs["rsi"],
                                              grabs["bb"], states)
                        self.save_region_overlay()
                    elif key == 'q':
                        self.running = False
                        runtime = time.time() - start_run
                        minutes, seconds = divmod(runtime, 60)
                        log_content = (
                            f"\n[confluence] Time: {datetime.now().strftime('%H:%M')}  "
                            f"Date: {date.today()}\n"
                            f"Runtime: {int(minutes)} min {seconds:.2f} sec\n"
                            f"Frames: {self.frame_count} "
                            f"(avg {runtime / max(1, self.frame_count) * 1000:.0f} ms/frame)\n"
                            f"Final mode: {self.mode}\n"
                            f"Final number of buys: {self.buy_count}\n"
                            f"Final number of sells: {self.sell_count}\n"
                        )
                        print(log_content)
                        with open("log.txt", "a") as log_file:
                            log_file.write(log_content)
                        break

                    time.sleep(0.05)

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting...")
            finally:
                self.finished.emit()


class MarketWorker:
    def __init__(self, trade_mode):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.detection_thread = ConfluenceWorker(trade_mode)
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
