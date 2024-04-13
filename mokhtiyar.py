# Trading on change of colors in a band
import threading
import traceback
import numpy as np
import pyautogui
from PIL import ImageDraw, Image
import keyboard
import time
import pyttsx3
from utils import Utils

class PIXEL_TRADER:

    def __init__(self):
        self.utils = Utils()
        # self.utils.green_band = [174, 255, 50]
        # self.utils.green_bg = (161, 187, 137)
        # self.utils.purple_bg = (166, 160, 180)
        # self.utils.purple_band = [150, 4, 212]
        # self.utils.GREEN_STATE = 0
        # self.utils.PURPLE_STATE = 1
        # self.utils.STATUS = None

    # def self.utils.speak(self, text):
    #     memory_thread = threading.Thread(target=self.utils.speak_thread, args=[text])
    #     memory_thread.start()

    # def self.utils.speak_thread(self, text):
    #     engine = pyttsx3.init()
    #     engine.say(text)
    #     engine.runAndWait()

    # def self.utils.draw_circle(self, image, x, y, radius=10):
    #     draw = ImageDraw.Draw(image)
    #     x1 = x - radius
    #     y1 = y - radius
    #     x2 = x + radius
    #     y2 = y + radius
    #     draw.ellipse((x1, y1, x2, y2), outline='red')
    #     image.show()

    # def self.utils.get_top_right(self, screenshot_array, color):
    #     mask = np.all(screenshot_array == color, axis=-1)
    #     indices = np.argwhere(mask)
    #     if len(indices) == 0:
    #         return [-1, -1]
    #     max_index = np.argmax(indices[:, 1])
    #     return indices[max_index]

    # def self.utils.get_pixel(self, screenshot_array):
    #     green_pt = self.utils.get_top_right(screenshot_array, list(self.utils.green_bg))
    #     purple_pt = self.utils.get_top_right(screenshot_array, list(self.utils.purple_bg))
    #     if green_pt[1] > purple_pt[1]:
    #         self.utils.STATUS = self.utils.GREEN_STATE
    #         return green_pt
    #     else:
    #         self.utils.STATUS = self.utils.PURPLE_STATE
    #         return purple_pt

    # def self.utils.get_coordinates(self):
    #     point = pyautogui.position()
    #     print(point)

    # def self.utils.buy(self):
    #     pyautogui.click(x=1798, y=42)

    # def self.utils.sell(self):
    #     pyautogui.click(x=1876, y=110)

    # def self.utils.reverse(self):
    #     pyautogui.click(x=1798, y=145)

    # def self.utils.close(self):
    #     pyautogui.click(x=1876, y=142)

    def buy_or_sell(self):
        while True:
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.utils.green_bg and self.utils.STATUS == self.utils.PURPLE_STATE:
                self.utils.STATUS = self.utils.GREEN_STATE
                self.utils.buy()
                break
            elif px == self.utils.purple_bg and self.utils.STATUS == self.utils.GREEN_STATE:
                self.utils.STATUS = self.utils.PURPLE_STATE
                self.utils.sell()
                break

    def reverse_after_buy_or_sell(self):
        while True:
            if self.last_keystroke == 'down' or self.last_keystroke == 'up' or self.last_keystroke == 'enter':
                return
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.utils.green_bg and self.utils.STATUS == self.utils.PURPLE_STATE:
                self.utils.STATUS = self.utils.GREEN_STATE
                self.utils.reverse()
            elif px == self.utils.purple_bg and self.utils.STATUS == self.utils.GREEN_STATE:
                self.utils.STATUS = self.utils.PURPLE_STATE
                self.utils.reverse()

    def initial_setup(self):
            screen_width, screen_height = pyautogui.size()
            X_MARGIN_LEFT = 0 #screen_width - 445
            Y_TOP_MARGIN = 0
            X_MARGIN_RIGHT = screen_width - 245
            Y_BOTTOM_MARGIN = screen_height - 50
            pyautogui.hotkey('alt', 'tab')
            screenshot = np.array(pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN]))
            self.pixel_pt = self.utils.get_pixel(screenshot)
            # self.pixel_pt = [int(self.pixel_pt[1]), int(self.pixel_pt[0])]
            print('Pixel Point:', self.pixel_pt)

    def run_buy(self):
        while True:
            if self.last_keystroke == 'down' or self.last_keystroke == 'right':
                return
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.utils.green_bg and STATUS == self.utils.PURPLE_STATE:
                STATUS = self.utils.GREEN_STATE
                self.utils.buy()
            elif px == self.utils.purple_bg and STATUS == self.utils.GREEN_STATE:
                STATUS = self.utils.PURPLE_STATE
                self.utils.close()

    def run_sell(self):
        while True:
            if self.last_keystroke == 'up' or self.last_keystroke == 'right':
                return
            px = pyautogui.pixel(int(self.pixel_pt[1]), int(self.pixel_pt[0]))
            if px == self.utils.purple_bg and STATUS == self.utils.GREEN_STATE:
                STATUS = self.utils.PURPLE_STATE
                self.utils.sell()
            elif px == self.utils.green_bg and STATUS == self.utils.PURPLE_STATE:
                STATUS = self.utils.GREEN_STATE
                self.utils.close()

    def run_buy_or_sell(self):
        self.buy_or_sell()
        self.reverse_after_buy_or_sell()

    def handle_thread(self, function):
        for event in self.stop_events:
            event.set()
        self.stop_events.clear()
        stop_event = threading.Event()
        self.utils.buy_thread = threading.Thread(target=function)
        self.utils.buy_thread.start()
        self.utils.buy_thread.__st
        self.stop_events.append(stop_event)

    def key_thread(self):
        while True:
            keystroke = keyboard.read_key()  # This will block until a key is pressed
            print(f'Key pressed: {keystroke}')
            self.utils.speak(keystroke)
            self.last_keystroke = keystroke

    def keystroke_listener(self):
        self.last_keystroke = 'right'
        self.key_thread = threading.Thread(target=self.utils.get_coordinates)
        self.key_thread.start()
        while True:
            keystroke = keyboard.read_key()  # This will block until a key is pressed
            print(f'Key pressed: {keystroke}')
            if keystroke == 'enter':
                # self.utils.speak('Enter')
                print('Closed by user')
                self.utils.close()
                if self.last_keystroke == 'right':
                    self.utils.speak(self.run_buy_or_sell())
                break
            elif keystroke == 'up':
                # self.utils.speak('Up')
                self.utils.speak(self.run_buy())
                break
            elif keystroke == 'down':
                # self.utils.speak('Down')
                self.utils.speak(self.run_sell())
                break
            elif keystroke == 'right':
                # self.utils.speak('Right')
                self.utils.speak(self.run_buy_or_sell())
                break
            elif keystroke == 'esc':
                self.utils.speak('Escape')
                break
            self.last_keystroke = keystroke

    def keystroke_listener_TBD(self):
        self.last_keystroke = 'a'
        self.key_thread = threading.Thread(target=self.utils.get_coordinates)
        # self.stop_events = []
        # stop_event = threading.Event()
        # self.utils.buy_thread = threading.Thread(target=self.run_buy_or_sell)
        # self.utils.buy_thread.start()
        # self.stop_events.append(stop_event)
        while True:
            keystroke = keyboard.read_key()  # This will block until a key is pressed
            print(f'Key pressed: {keystroke}')
            if keystroke == 'enter':
                self.utils.speak('Enter')
                # print('Closed by user')
                # self.utils.close()
                # if self.last_keystroke == 'right':
                #     self.handle_thread(self.run_buy_or_sell)
                # break
            elif keystroke == 'up':
                self.utils.speak('Up')
                # self.handle_thread(self.run_buy)
                # break
            elif keystroke == 'down':
                self.utils.speak('Down')
                # self.handle_thread(self.run_sell)
                # break
            elif keystroke == 'right':
                self.utils.speak('Right')
                # self.handle_thread(self.run_buy_or_sell)
                # break
            elif keystroke == 'esc':
                self.utils.speak('Escape')
                break
            self.last_keystroke = keystroke

    def run(self):
        try:
            # self.initial_setup()
            self.keystroke_listener()
        except Exception as e:
            self.utils.close()
            print('ERROR ::', e)
            print(traceback.format_exc())

    def test(self):
        self.initial_setup()
        self.buy_or_sell()
        self.reverse_after_buy_or_sell()

if __name__ == '__main__':
    trader = PIXEL_TRADER()
    trader.run()