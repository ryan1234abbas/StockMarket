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

# Speed up pyautogui: default is a 0.1s pause after EVERY call, which adds
# ~0.3s to a Close+Buy/Sell sequence
pyautogui.PAUSE = 0.01

# =====================================================================
# market_worker6.py
# Simple color-based trader driven by the 3020 chart (right chart):
#   Rightmost background AREA turns GREEN (bullish) -> Close, then Buy Mkt
#   Rightmost background AREA turns RED/pink (bearish) -> Close, then Sell Mkt
# The signal is the SuperAreas background shading band at the right edge
# of the chart - it matches the Bullish/Bearish text (which renders in
# black and cannot be read by color). Individual brick colors are NOT
# used: bricks zig-zag red/green constantly within a single trend phase.
# =====================================================================

# Plot area of the 3020 chart (right chart) as ratios of the FULL screen:
# x1, y1, x2, y2. Starts below the indicator header text and ends before
# the price axis, so only bricks/overlays are inside.
BRICK_REGION = {
    "Windows": (0.47, 0.18, 0.825, 0.92),
    "Darwin":  (0.47, 0.18, 0.825, 0.92),
}

# DOM button centers as ratios of screen width/height (measured from the
# trading layout screenshots). If you save screenshots of the buttons as
# buy_mkt.png / sell_mkt.png / close.png in this folder, the program will
# locate them on screen automatically and ignore these ratios.
BUY_BUTTON_RATIO = (0.905, 0.049)
SELL_BUTTON_RATIO = (0.967, 0.049)
CLOSE_BUTTON_RATIO = (0.967, 0.174)
BUY_BUTTON_IMG = "buy_mkt.png"
SELL_BUTTON_IMG = "sell_mkt.png"
CLOSE_BUTTON_IMG = "close.png"

POLL_INTERVAL = 0.03       # seconds between screen checks (~30 checks/sec)
STABLE_FRAMES = 3          # consecutive identical readings before state is trusted
TRADE_ON_FIRST_STATE = False  # wait for the first color CHANGE before trading
CLOSE_BEFORE_TRADE_DELAY = 0.1  # seconds between clicking Close and the new order
STATUS_EVERY_N_FRAMES = 100  # console status line interval (printing is slow)
EDGE_GRAB_WIDTH_RATIO = 0.06  # only this slice of screen width, ending at the
                              # chart's right edge, is captured each frame
STRIP_WIDTH = 25           # pixel columns at the right edge to sample
MIN_AREA_PIXELS = 150      # background pixels needed to accept a reading
DOMINANCE_RATIO = 1.5      # winning color needs this x more pixels


def classify_rightmost_area(crop_bgr):
    """Return ('GREEN'|'RED'|None, debug_string) for the background shading
    band at the right edge of a BGR crop of the 3020 chart plot area.

    The SuperAreas shading is pale (low-to-moderate saturation, bright),
    which separates it from everything drawn on top of it: bricks and
    marker boxes are vivid (high saturation), the gray margin/axis has
    near-zero saturation, and text/lines are dark or colorless."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, "empty crop"

    hsv = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2HSV)

    # Background shading only: pale but not gray, and bright
    pink1 = cv2.inRange(hsv, (0, 15, 110), (15, 95, 255))
    pink2 = cv2.inRange(hsv, (160, 15, 110), (180, 95, 255))
    pink = cv2.bitwise_or(pink1, pink2)
    sage = cv2.inRange(hsv, (35, 15, 110), (85, 95, 255))

    # Find the rightmost column that actually contains area shading, so the
    # gray margin/price axis at the crop's right edge is skipped automatically
    pink_cols = (pink > 0).sum(axis=0)
    sage_cols = (sage > 0).sum(axis=0)
    h = crop_bgr.shape[0]
    shaded = np.where((pink_cols + sage_cols) >= h * 0.3)[0]
    if shaded.size == 0:
        return None, "no shaded columns found"

    x_last = int(shaded[-1])
    window = slice(max(0, x_last - STRIP_WIDTH + 1), x_last + 1)
    r = int(pink_cols[window].sum())
    g = int(sage_cols[window].sum())

    debug = f"edge_x={x_last} pink_px={r} green_px={g}"
    if g >= MIN_AREA_PIXELS and g > r * DOMINANCE_RATIO:
        return "GREEN", debug
    if r >= MIN_AREA_PIXELS and r > g * DOMINANCE_RATIO:
        return "RED", debug
    return None, debug + " (no clear area)"


class BrickWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, trade_mode):
        super().__init__()
        self.mode = trade_mode
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.buy_count = 0
        self.sell_count = 0

        # Signal state machine
        self.confirmed_state = None
        self.candidate_state = None
        self.candidate_frames = 0

        # Cached button positions (found once, reused)
        self.button_pos = {}

    def find_button(self, which):
        """Locate a DOM button center on screen. Template image first,
        fixed screen-ratio fallback second. Result is cached."""
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

    def close_position(self):
        """Click the DOM Close button to flatten the existing contract."""
        x, y = self.find_button("CLOSE")
        pyautogui.click(x, y)
        print("Clicked CLOSE to flatten existing position")
        time.sleep(CLOSE_BEFORE_TRADE_DELAY)

    def execute_trade(self, new_state):
        """Close any open contract, then click the market order button
        for the newly confirmed brick color."""
        if new_state == "GREEN" and self.mode in ("buy", "both"):
            self.close_position()
            x, y = self.find_button("BUY")
            pyautogui.click(x, y)
            self.buy_count += 1
            return "BUY"

        elif new_state == "RED" and self.mode in ("sell", "both"):
            self.close_position()
            x, y = self.find_button("SELL")
            pyautogui.click(x, y)
            self.sell_count += 1
            return "SELL"

        return None

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

        start_run = time.time()

        # Locate all buttons up front: the image search can take seconds and
        # must not happen while a trade is waiting to be placed
        print("Locating DOM buttons...")
        for which in ("BUY", "SELL", "CLOSE"):
            self.find_button(which)

        print("Keys (console window must be focused): "
              "p=pause/resume  b=buy only  s=sell only  a=both  d=debug dump  q=quit")

        with mss.mss() as sct:
            # Grab ONLY a narrow strip at the chart's right edge - the
            # classifier just needs the rightmost shaded columns, and a small
            # capture keeps the loop running at ~30 checks/sec
            mon = sct.monitors[1]
            x1r, y1r, x2r, y2r = BRICK_REGION.get(platform.system(), BRICK_REGION["Windows"])
            strip_x1r = max(x1r, x2r - EDGE_GRAB_WIDTH_RATIO)
            grab_region = {
                "left": mon["left"] + int(mon["width"] * strip_x1r),
                "top": mon["top"] + int(mon["height"] * y1r),
                "width": int(mon["width"] * (x2r - strip_x1r)),
                "height": int(mon["height"] * (y2r - y1r)),
                "mon": 1,
            }

            try:
                while self.running:
                    loop_start = time.time()
                    crop = np.array(sct.grab(grab_region))[:, :, :3]

                    # Save the crop once at startup so the region can be verified
                    if self.frame_count == 0:
                        cv2.imwrite("brick_debug.png", crop)
                        print("Saved right-edge strip to brick_debug.png - open it "
                              "and check it shows the 3020 chart's right edge "
                              "(newest bricks + background shading)")

                    raw_state, debug = classify_rightmost_area(crop)

                    # Debounce: require STABLE_FRAMES identical readings
                    if raw_state == self.candidate_state:
                        self.candidate_frames += 1
                    else:
                        self.candidate_state = raw_state
                        self.candidate_frames = 1

                    decision = None
                    if (self.candidate_state is not None and
                            self.candidate_frames >= STABLE_FRAMES and
                            self.candidate_state != self.confirmed_state):

                        previous = self.confirmed_state
                        self.confirmed_state = self.candidate_state
                        print(f"AREA CONFIRMED: {self.confirmed_state} "
                              f"(was {previous or 'unknown'})")

                        if self.paused:
                            print("PAUSED - trade skipped")
                        elif previous is not None or TRADE_ON_FIRST_STATE:
                            decision = self.execute_trade(self.confirmed_state)
                        else:
                            print("Startup state - waiting for an area color CHANGE")

                    if decision:
                        print(f"Trade decision: {decision}")

                    self.frame_count += 1
                    if decision or self.frame_count % STATUS_EVERY_N_FRAMES == 0:
                        print(f"Frame {self.frame_count}: area={raw_state or 'unknown'} "
                              f"confirmed={self.confirmed_state or 'unknown'} "
                              f"[{debug}] mode={self.mode}"
                              f"{' PAUSED' if self.paused else ''} "
                              f"buys={self.buy_count} sells={self.sell_count}")

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
                        cv2.imwrite("brick_debug.png", crop)
                        print("Dumped current crop to brick_debug.png")
                    elif key == 'q':
                        self.running = False
                        print("\nQ PRESSED...STOPPING PROGRAM...")
                        runtime = time.time() - start_run
                        minutes, seconds = divmod(runtime, 60)

                        log_content = (
                            f"\n[worker6] Time: {datetime.now().strftime('%H:%M')}  "
                            f"Date: {date.today()}\n"
                            f"Runtime: {int(minutes)} min {seconds:.2f} sec\n"
                            f"Frames: {self.frame_count} "
                            f"(avg {runtime / max(1, self.frame_count) * 1000:.0f} ms/frame)\n"
                            f"Final area state: {self.confirmed_state or 'unknown'}\n"
                            f"Final mode: {self.mode}\n"
                            f"Final number of buys: {self.buy_count}\n"
                            f"Final number of sells: {self.sell_count}\n"
                        )
                        print(log_content)
                        with open("log.txt", "a") as log_file:
                            log_file.write(log_content)
                        break

                    # Adaptive sleep: subtract time already spent this loop
                    elapsed = time.time() - loop_start
                    if elapsed < POLL_INTERVAL:
                        time.sleep(POLL_INTERVAL - elapsed)

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, exiting...")
            finally:
                self.finished.emit()


class MarketWorker:
    def __init__(self, trade_mode):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.detection_thread = BrickWorker(trade_mode)
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
