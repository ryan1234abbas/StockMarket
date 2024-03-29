# Description: This file contains the utility functions for the bot.
import threading
import numpy as np
import pyautogui
from PIL import ImageDraw, Image
import pyttsx3


class Utils:

    def __init__(self):
        self.green_band = (174, 255, 50)
        self.green_bg = (161, 187, 137)
        self.purple_bg = (166, 160, 180)
        self.purple_band = (150, 4, 212)
        self.green_long_trade = (4, 102, 4)
        self.green_small_trade = (53, 206, 53)
        self.purple_long_trade = (130, 4, 130)
        self.purple_small_trade = (255, 4, 255)
        self.GREEN_STATE = 0
        self.PURPLE_STATE = 1
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

    def get_top_right_x(self, screenshot_array, color):
        mask = np.all(screenshot_array == color, axis=-1)
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return -1
        max_index = np.argmax(indices[:, 1])
        return indices[max_index][1]

    def get_top_right(self, screenshot_array, color):
        mask = np.all(screenshot_array == color, axis=-1)
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return [-1, -1]
        max_index = np.argmax(indices[:, 1])
        return indices[max_index]

    def get_top_n_bottom_pixel(self, screenshot_array, color):
        mask = np.all(screenshot_array == color, axis=-1)
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return [-1, -1]
        max_index = np.argmax(indices[:, 1])
        return indices[max_index], indices[-1]

    def get_right_edge(self, screenshot_array):
        green_top, green_bottom = self.get_top_right(screenshot_array, list(self.green_bg))
        purple_top, purple_bottom = self.get_top_right(screenshot_array, list(self.purple_bg))
        if green_top[1] > purple_top[1]:
            self.STATUS = self.GREEN_STATE
            return green_top, green_bottom
        else:
            self.STATUS = self.PURPLE_STATE
            return purple_top, purple_bottom

    def get_pixel(self, screenshot_array):
        green_pt = self.get_top_right(screenshot_array, list(self.green_bg))
        purple_pt = self.get_top_right(screenshot_array, list(self.purple_bg))
        if green_pt[1] > purple_pt[1]:
            self.STATUS = self.GREEN_STATE
            return green_pt
        else:
            self.STATUS = self.PURPLE_STATE
            return purple_pt

    def get_coordinates():
        point = pyautogui.position()
        print(point)

    def buy():
        pyautogui.click(x=1798, y=42)

    def sell():
        pyautogui.click(x=1876, y=110)

    def reverse():
        pyautogui.click(x=1798, y=145)

    def close():
        pyautogui.click(x=1876, y=142)
