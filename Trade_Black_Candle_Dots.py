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
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH, self.top_pixel[0], self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0])))
            up_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.up_pin_point)
            down_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.down_pin_point)
            # screenshot = Image.fromarray(screenshot_array)
            # self.utils.draw_circle(screenshot, int(up_pin_pt[1]), int(up_pin_pt[0]))
            # self.utils.draw_circle(screenshot, int(down_pin_pt[1]), int(down_pin_pt[0]))
            # print(up_pin_pt, down_pin_pt)
            if up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS == self.utils.PURPLE_STATE:
                # self.utils.speak('Green')
                self.utils.close()
                self.utils.buy()
                # self.utils.reverse()
                self.utils.STATUS = self.utils.GREEN_STATE
                # if up_pin_pt[0] > self.previous_up_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_up_pin_pt = up_pin_pt
            elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS == self.utils.GREEN_STATE:
                # self.utils.speak('Cyan')
                self.utils.close()
                self.utils.sell()
                # self.utils.reverse()
                self.utils.STATUS = self.utils.PURPLE_STATE
                # if down_pin_pt[0] > self.previous_down_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_down_pin_pt = down_pin_pt
            elif up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS is None:
                # self.utils.speak('Green')
                self.utils.buy()
                self.utils.STATUS = self.utils.GREEN_STATE
                # if up_pin_pt[0] > self.previous_up_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_up_pin_pt = up_pin_pt
            elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS is None:
                # self.utils.speak('Cyan')
                self.utils.sell()
                self.utils.STATUS = self.utils.PURPLE_STATE
                # if down_pin_pt[0] > self.previous_down_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_down_pin_pt = down_pin_pt


if __name__ == '__main__':
    pandu = Pandu()
    pandu.run()