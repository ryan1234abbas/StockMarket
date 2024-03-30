import sys
import pyautogui
import utils
import numpy as np

class ParameterizedTrade:

    def __init__(self, strategy, colors_to_check):
        self.EDGE_DELTA = 5
        self.utils = utils.Utils()
        self.green_lines_list = []
        for color in colors_to_check:
            if color == 'mb':
                self.green_lines_list.append(self.utils.green_band)
            elif color == 'st':
                self.green_lines_list.append(self.utils.green_short_trigger)
            elif color == 'lt':
                self.green_lines_list.append(self.utils.green_long_trigger)
            elif color == 'bg':
                self.green_lines_list.append(self.utils.green_bg)
                
        self.purple_lines_list = []
        for color in colors_to_check:
            if color == 'mb':
                self.purple_lines_list.append(self.utils.purple_band)
            elif color == 'st':
                self.purple_lines_list.append(self.utils.purple_short_trigger)
            elif color == 'lt':
                self.purple_lines_list.append(self.utils.purple_long_trigger)
            elif color == 'bg':
                self.purple_lines_list.append(self.utils.purple_bg)

        self.run_initial_setup()
        if strategy == 'buy&sell':
            self.run_buy_sell()
        elif strategy == 'buy':
            self.run_buy()
        elif strategy == 'sell':
            self.run_sell()

    def run_initial_setup(self):
        """Get screenshot, find the right edge of the green and purple background, and set the status accordingly."""
        # pyautogui screenshot region: 
        # The box to capture. Default is the entire screen. If a four-integer tuple is passed, it is interpreted as the left, top, width, and height of the region to capture.    
        # left: The x-coordinate of the top-left corner of the region to capture.
        # top: The y-coordinate of the top-left corner of the region to capture.
        # width: The width of the region to capture.
        # height: The height of the region to capture.
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

    def run_buy_sell(self):
        print('Running buy_and_sell')
        screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
        green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
        purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
        if all(green_list):
            self.utils.STATUS = self.utils.GREEN_STATE
            while True:
                screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                if any(purple_list):
                    break
        elif all(purple_list):
            self.utils.STATUS = self.utils.PURPLE_STATE
            while True:
                screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                if any(green_list):
                    break
        while True:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if all(green_list):
                self.utils.STATUS = self.utils.GREEN_STATE
                self.utils.buy()
                self.close_green()
            elif all(purple_list):
                self.utils.STATUS = self.utils.PURPLE_STATE
                self.utils.sell()
                self.close_purple()
    
    def close_green(self):
        while True:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if any(purple_list):
                self.utils.close()
                return

    def close_purple(self):
        while True:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            if any(green_list):
                self.utils.close()
                return

    def run_buy(self):
        print('Running buy')
        screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
        green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
        if all(green_list):
            self.utils.STATUS = self.utils.GREEN_STATE
            while True:
                screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                if any(purple_list):
                    break
        while True:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            if all(green_list):
                self.utils.STATUS = self.utils.GREEN_STATE
                self.utils.buy()
                self.close_green()
    
    def run_sell(self):
        print('Running sell')
        screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
        purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
        if all(purple_list):
            self.utils.STATUS = self.utils.PURPLE_STATE
            while True:
                screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                if any(green_list):
                    break
        while True:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if all(purple_list):
                self.utils.STATUS = self.utils.PURPLE_STATE
                self.utils.sell()
                self.close_purple()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python parameterized_trade.py <strategy> <colors_to_check>')
        print('strategy options: "buy&sell", "buy", "sell"')
        print('colors_to_check options: Any combo of: "mb", "st", "lt", "bg"')
        print('Example: py parameterized_trade.py "buy&sell" "mb", "st", "lt"')
        sys.exit(1)
    
    strategy = sys.argv[1] # buy&sell, buy, sell
    colors_to_check = sys.argv[2:] # All arguments from argv[2] onwards

    if strategy not in ['buy&sell', 'buy', 'sell'] or any(line not in ['mb', 'st', 'lt', 'bg'] for line in colors_to_check):
        print('Invalid strategy or colors_to_check argument.')
        print('Usage: python parameterized_trade.py <strategy> <colors_to_check>')
        print('strategy options: "buy&sell", "buy", "sell"')
        print('colors_to_check options: Any combo of: "mb", "st", "lt", "bg"')
        print('Example: py parameterized_trade.py "buy&sell" "mb", "st", "lt"')
        sys.exit(1)

    trader = ParameterizedTrade(strategy, colors_to_check)
