# Description: This file contains the utility functions for the bot.
import threading
import numpy as np
import pyautogui
from PIL import ImageDraw, Image
import pyttsx3


class Utils:

    def __init__(self):
        self.green_bg = (161, 187, 137) # LawnGreen
        self.purple_bg = (166, 160, 180) # MediumPurple
        self.green_band = (174, 255, 50) # GreenYellow
        self.purple_band = (150, 4, 212) # DarkViolet
        self.green_long_trigger = (4, 102, 4) # DarkGreen
        self.green_short_trigger = (53, 206, 53) # LimeGreen
        self.purple_long_trigger = (141, 4, 141) # DarkMagenta
        self.purple_short_trigger = (255, 4, 255) # Magenta
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

    def get_top_and_bottom_pixel(self, screenshot_array, color):
        mask = np.all(screenshot_array == color, axis=-1)
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return [-1, -1], [-1, -1]
        max_index = np.argmax(indices[:, 1])
        return indices[max_index], indices[-1]

    def get_right_edge(self, screenshot_array):
        green_top, green_bottom = self.get_top_and_bottom_pixel(screenshot_array, list(self.green_bg))
        purple_top, purple_bottom = self.get_top_and_bottom_pixel(screenshot_array, list(self.purple_bg))
        if green_top[1] > purple_top[1]:
            self.STATUS = self.GREEN_STATE
            return green_top, green_bottom
        else:
            self.STATUS = self.PURPLE_STATE
            return purple_top, purple_bottom

    def check_color_in_vertical_line(self, screenshot_array, top, bottom, color):
        line = screenshot_array[top[0]:bottom[0]+1, top[1]:top[1]+1]
        return np.any(np.all(line == color, axis=-1))

    def check_color_in_all_pixels(self, screenshot_array, color):
        """ Check if the color is present in any pixel from screenshot."""
        mask = np.all(screenshot_array == color, axis=-1)
        # self.draw_circle(Image.fromarray(screenshot_array), mask[0], mask[1])
        return np.any(mask)

    def get_pixel(self, screenshot_array):
        green_pt = self.get_top_right(screenshot_array, list(self.green_bg))
        purple_pt = self.get_top_right(screenshot_array, list(self.purple_bg))
        if green_pt[1] > purple_pt[1]:
            self.STATUS = self.GREEN_STATE
            return green_pt
        else:
            self.STATUS = self.PURPLE_STATE
            return purple_pt

    def get_coordinates(self):
        point = pyautogui.position()
        print(point)

    def buy(self):
        pyautogui.click(x=1803, y=43)

    def sell(self):
        pyautogui.click(x=1879, y=42)

    def reverse(self):
        pyautogui.click(x=1802, y=144)

    def close(self):
        pyautogui.click(x=1877, y=144)

if __name__ == '__main__':
    utils = Utils()
    utils.get_coordinates()