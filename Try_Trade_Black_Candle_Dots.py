import numpy as np
import pyautogui
import utils
from PIL import Image

class Pandu:
    def __init__(self) -> None:
        self.utils = utils.Utils()
        self.EDGE_DELTA = 5
        self.WIDTH = 20

    def initial_setup(self):
        screen_width, screen_height = pyautogui.size()
        X_TOP = 0 #screen_width - 445
        Y_TOP = 0
        WIDTH = screen_width - 245
        HEIGHT = screen_height - 50
        pyautogui.hotkey('alt', 'tab')
        screenshot_array = np.array(pyautogui.screenshot(region=[X_TOP, Y_TOP, WIDTH, HEIGHT]))
        self.top_pixel, self.bottom_pixel = self.utils.get_right_edge(screenshot_array)
        self.top_pixel = list(map(int, self.top_pixel))
        self.bottom_pixel = list(map(int, self.bottom_pixel))

    def run(self):
        self.initial_setup()
        self.previous_up_pin_pt = [-1, -1]
        self.previous_down_pin_pt = [-1, -1]
        self.utils.STATUS = None
        while True:
            ocr_image = pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH + 5, self.top_pixel[0], 2*self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0]))
            screenshot_array = np.array(ocr_image)
            up_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.up_pin_point)
            down_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.down_pin_point)
            self.utils.CURRENT_CLASS = self.utils.classify(ocr_image)
            if up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS == self.utils.PURPLE_STATE:
                if self.utils.CURRENT_CLASS == self.utils.PREVIOUS_CLASS:
                    continue
                print('Previous:', self.utils.PREVIOUS_CLASS)
                print('Current:', self.utils.CURRENT_CLASS)
                if ((self.utils.CURRENT_CLASS == 'HL' and self.utils.PREVIOUS_CLASS == 'HH') or (self.utils.CURRENT_CLASS == 'HL' and self.utils.PREVIOUS_CLASS == 'LH')):
                    print("Buy")
                    self.utils.buy()
                print()
                self.utils.STATUS = self.utils.GREEN_STATE
                self.utils.PREVIOUS_CLASS = self.utils.CURRENT_CLASS
            elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS == self.utils.GREEN_STATE:
                if self.utils.CURRENT_CLASS == self.utils.PREVIOUS_CLASS:
                    continue
                print('Previous:', self.utils.PREVIOUS_CLASS)
                print('Current:', self.utils.CURRENT_CLASS)
                if ((self.utils.CURRENT_CLASS == 'LH' and self.utils.PREVIOUS_CLASS == 'LL') or (self.utils.CURRENT_CLASS == 'LH' and self.utils.PREVIOUS_CLASS == 'HL')):
                    print("sell")
                    self.utils.sell()
                print()
                self.utils.STATUS = self.utils.PURPLE_STATE
                self.utils.PREVIOUS_CLASS = self.utils.CURRENT_CLASS
            elif up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS is None:
                print(self.utils.CURRENT_CLASS)
                self.utils.STATUS = self.utils.GREEN_STATE
            elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS is None:
                print(self.utils.CURRENT_CLASS)
                self.utils.STATUS = self.utils.PURPLE_STATE


if __name__ == '__main__':
    pandu = Pandu()
    pandu.run()