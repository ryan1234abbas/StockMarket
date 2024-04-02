# Trading on change of colors in a band
import threading
import traceback
import numpy as np
import pyautogui
from PIL import ImageDraw, Image
import keyboard
import time
import pyttsx3

class PIXEL_TRADER:

    def __init__(self):
        self.green_band = [174, 255, 50]
        self.green_bg = (161, 187, 137)
        self.purple_bg = (166, 160, 180)
        self.purple_band = [150, 4, 212]
        self.GREEN = 0
        self.PURPLE = 1
        self.STATUS = None

    def speak(self, text):
        memory_thread = threading.Thread(target=self.speak_thread, args=[text])
        memory_thread.start()

    def speak_thread(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def draw_circle(self, image, x, y, radius=10):
        draw = ImageDraw.Draw(image)
        x1 = x - radius
        y1 = y - radius
        x2 = x + radius
        y2 = y + radius
        draw.ellipse((x1, y1, x2, y2), outline='red')
        image.show()

    def get_top_right(self, screenshot_array, color):
        mask = np.all(screenshot_array == color, axis=-1)
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return [-1, -1]
        max_index = np.argmax(indices[:, 1])
        return indices[max_index]

    def get_pixel(self, screenshot_array):
        green_pt = self.get_top_right(screenshot_array, list(self.green_bg))
        purple_pt = self.get_top_right(screenshot_array, list(self.purple_bg))
        if green_pt[1] > purple_pt[1]:
            self.STATUS = self.GREEN
            return green_pt
        else:
            self.STATUS = self.PURPLE
            return purple_pt

    def get_coordinates(self):
        point = pyautogui.position()
        print(point)

    def buy(self):
        pyautogui.click(x=1798, y=42)

    def sell(self):
        pyautogui.click(x=1876, y=110)

    def reverse(self):
        pyautogui.click(x=1798, y=145)

    def close(self):
        pyautogui.click(x=1876, y=142)

    def buy_or_sell(self):
        while True:
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.green_bg and self.STATUS == self.PURPLE:
                self.STATUS = self.GREEN
                self.buy()
                break
            elif px == self.purple_bg and self.STATUS == self.GREEN:
                self.STATUS = self.PURPLE
                self.sell()
                break

    def reverse_after_buy_or_sell(self):
            while True:
                px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
                if px == self.green_bg and self.STATUS == self.PURPLE:
                    self.STATUS = self.GREEN
                    self.reverse()
                elif px == self.purple_bg and self.STATUS == self.GREEN:
                    self.STATUS = self.PURPLE
                    self.reverse()

    def initial_setup(self):
            screen_width, screen_height = pyautogui.size()
            X_MARGIN_LEFT = 0 #screen_width - 445
            Y_TOP_MARGIN = 0
            X_MARGIN_RIGHT = screen_width - 245
            Y_BOTTOM_MARGIN = screen_height - 50
            pyautogui.hotkey('alt', 'tab')
            screenshot = np.array(pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN]))
            self.pixel_pt = self.get_pixel(screenshot)
            # self.pixel_pt = [int(self.pixel_pt[1]), int(self.pixel_pt[0])]
            print('Pixel Point:', self.pixel_pt)

    def run_buy(self):
        while True:
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.green_bg and STATUS == self.PURPLE:
                STATUS = self.GREEN
                self.buy()
            elif px == self.purple_bg and STATUS == self.GREEN:
                STATUS = self.PURPLE
                self.close()

    def run_sell(self):
        while True:
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.purple_bg and STATUS == self.GREEN:
                STATUS = self.PURPLE
                self.sell()
            elif px == self.green_bg and STATUS == self.PURPLE:
                STATUS = self.GREEN
                self.close()

    def run_buy_or_sell(self):
        self.buy_or_sell()
        self.reverse_after_buy_or_sell()

    def handle_thread(self, function):
        for event in self.stop_events:
            event.set()
        self.stop_events.clear()
        stop_event = threading.Event()
        buy_thread = threading.Thread(target=function)
        buy_thread.start()
        self.stop_events.append(stop_event)

    def keystroke_listener(self):
        last_keystroke = 'a'
        self.stop_events = []
        stop_event = threading.Event()
        buy_thread = threading.Thread(target=self.run_buy_or_sell)
        buy_thread.start()
        self.stop_events.append(stop_event)
        while True:
            keystroke = keyboard.read_key()  # This will block until a key is pressed
            print(f'Key pressed: {keystroke}')
            if keystroke == 'enter':
                print('Closed by user')
                self.close()
                if last_keystroke == 'right':
                    self.handle_thread(self.run_buy_or_sell)
                break
            elif keystroke == 'up':
                self.handle_thread(self.run_buy)
                break
            elif keystroke == 'down':
                self.handle_thread(self.run_sell)
                break
            elif keystroke == 'right':
                self.handle_thread(self.run_buy_or_sell)
                break
            last_keystroke = keystroke

    def run(self):
        try:
            self.initial_setup()
            self.keystroke_listener()
        except Exception as e:
            self.close()
            print('ERROR ::', e)
            print(traceback.format_exc())

    def test(self):
        self.initial_setup()
        self.buy_or_sell()
        self.reverse_after_buy_or_sell()

if __name__ == '__main__':
    trader = PIXEL_TRADER()
    trader.run()