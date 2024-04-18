# Safe exit of threads: Press esc. Do not press right ctrl + C or right ctrl + X
# Keystroke will perform shortcuts defined in the active window along with the actions defined in the script.

############################################ DO NOT PRESS ANY KEY OTHER THAN THE DEFINED KEYS ############################################
# Keys:
# right shift: Buy
# right ctrl: Sell
# space: Buy or Sell
# delete: Close
# esc: Exit

# Trading on change of colors in a band
import sys
import threading
import traceback
import numpy as np
import pyautogui
import keyboard
import time
from utils import Utils

class PIXEL_TRADER:

    def __init__(self, strategy, colors_to_check):
        self.utils = Utils()
        self.EDGE_DELTA = 5
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

    def old_buy_or_sell(self):
        key = 'space'
        while self.keystroke == key:
            print('buy or sell')
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            # Green lines are present
            if any([self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]):
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    if any([self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]):
                        self.utils.sell()
                        self.utils.STATUS = self.utils.PURPLE_STATE
                        break
                break
            else:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    if any([self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]):
                        self.utils.buy()
                        self.utils.STATUS = self.utils.GREEN_STATE
                        break
                break

    def first_buy_and_sell(self):
        # # self.utils.speak_thread('First buy and sell')
        key = 'space'
        print('Running buy or sell')
        self.utils.STATUS = None

        # First order
        while self.keystroke == key and self.utils.STATUS is None:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if all(purple_list):
                # self.utils.speak('All purple')
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if any(green_list):
                        # self.utils.speak('Buy first')
                        self.utils.buy()
                        self.utils.STATUS = self.utils.GREEN_STATE
                        break
            elif all(green_list):
                # self.utils.speak('All green')
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if any(purple_list):
                        self.utils.speak('sell first')
                        self.utils.sell()
                        self.utils.STATUS = self.utils.PURPLE_STATE
                        break

    def subsequent_buy_and_sell(self):
        key = 'space'
        # Subsequent orders
        while self.keystroke == key:
            if self.utils.STATUS == self.utils.GREEN_STATE:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if all(purple_list):
                        self.utils.close()
                        self.utils.STATUS = None
                        self.utils.PREV_STATUS = self.utils.GREEN_STATE
                        break
            elif self.utils.STATUS == self.utils.PURPLE_STATE:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if all(green_list):
                        self.utils.close()
                        self.utils.STATUS = None
                        self.utils.PREV_STATUS = self.utils.PURPLE_STATE
                        break
            else:
                while self.keystroke == key:
                    if self.utils.PREV_STATUS == self.utils.PURPLE_STATE:
                        screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                        green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                        if any(green_list):
                            self.utils.buy()
                            self.utils.STATUS = self.utils.GREEN_STATE
                            # self.utils.PREV_STATUS = None
                            break
                    else:
                        screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                        purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                        if any(purple_list):
                            self.utils.sell()
                            self.utils.STATUS = self.utils.PURPLE_STATE
                            # self.utils.PREV_STATUS = None
                            break

    def first_buy(self):
        # # self.utils.speak_thread('Running buy')
        key = 'right shift'
        self.utils.STATUS = None
        
        # First order
        while self.keystroke == key and self.utils.STATUS is None:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if all(purple_list):
                self.utils.speak('All purple')
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if any(green_list):
                        self.utils.speak('Buy first')
                        self.utils.buy()
                        self.utils.STATUS = self.utils.GREEN_STATE
                        break
        
    def subsequent_buy(self):
        # self.utils.speak_thread('Running buy')
        key = 'right shift'
        # Subsequent orders
        while self.keystroke == key:
            if self.utils.STATUS == self.utils.GREEN_STATE:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if all(purple_list):
                        self.utils.close()
                        self.utils.STATUS = None
                        break
            else:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if any(green_list):
                        self.utils.buy()
                        self.utils.STATUS = self.utils.GREEN_STATE
                        break

    def first_sell(self):
        # self.utils.speak_thread('Running sell')
        key = 'right ctrl'
        self.utils.STATUS = None

        # First order
        while self.keystroke == key and self.utils.STATUS is None:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            if all(green_list):
                self.utils.speak('All green')
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if any(purple_list):
                        self.utils.speak('sell first')
                        self.utils.sell()
                        self.utils.STATUS = self.utils.PURPLE_STATE
                        break
        
    def subsequent_sell(self):
        # self.utils.speak_thread('Running sell')
        key = 'right ctrl'

        # Subsequent orders
        while self.keystroke == key:
            if self.utils.STATUS == self.utils.PURPLE_STATE:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if all(green_list):
                        self.utils.close()
                        self.utils.STATUS = None
                        break
            else:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if any(purple_list):
                        self.utils.sell()
                        self.utils.STATUS = self.utils.PURPLE_STATE
                        break



    def old_reverse_after_buy_or_sell(self):
        key = 'space'
        while self.keystroke == key:
            print('Reverse after buy or sell')
            # screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            if self.utils.STATUS == self.utils.GREEN_STATE:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    if any([self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]):
                        # self.utils.close()
                        # self.utils.sell()
                        self.utils.reverse()
                        self.utils.STATUS = self.utils.PURPLE_STATE
                        break
            else:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    if any([self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]):
                        # self.utils.close()
                        # self.utils.buy()
                        self.utils.reverse()
                        self.utils.STATUS = self.utils.GREEN_STATE
                        break

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
       
    def close_green(self, key):
        while self.keystroke == key:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if all(purple_list):
                self.utils.close()
                return

    def close_purple(self, key):
        while self.keystroke == key:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            if all(green_list):
                self.utils.close()
                return

    def old_run_buy(self):
        key = 'right shift'
        while self.keystroke == key:
            print('Running buy')
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
            if any(green_list):
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if any(purple_list):
                        break
            else:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if all(green_list):
                        self.utils.buy()
                        self.close_green(key)

    def old_run_sell(self):
        key = 'right ctrl'
        while self.keystroke == key:
            print('Running sell')
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
            purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
            if any(purple_list):
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    green_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.green_lines_list]
                    if any(green_list):
                        break
            else:
                while self.keystroke == key:
                    screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1]-self.EDGE_DELTA, self.top_pixel[0], 1, self.bottom_pixel[0] - self.top_pixel[0])))
                    purple_list = [self.utils.check_color_in_all_pixels(screenshot_array, color) for color in self.purple_lines_list]
                    if all(purple_list):
                        self.utils.sell()
                        self.close_purple(key)

    def keystroke_thread(self):
        while True:
            self.keystroke = keyboard.read_key()  # This will block until a key is pressed
            # print(self.keystroke)
            if self.keystroke == 'esc':
                sys.exit()

    def keystroke_listener(self):
        self.last_keystroke = None
        self.keystroke = None
        self.key_thread = threading.Thread(target=self.keystroke_thread)
        self.key_thread.start()
        while True:
            time.sleep(0.01)
            if self.keystroke != self.last_keystroke:
                print(self.last_keystroke, self.keystroke)
                if self.keystroke == 'delete':
                    self.utils.close()
                    if self.last_keystroke == 'space':
                        self.keystroke = 'space'
                        self.subsequent_buy_and_sell()
                    elif self.last_keystroke == 'right shift':
                        self.keystroke = 'right shift'
                        self.subsequent_buy()
                    elif self.last_keystroke == 'right ctrl':
                        self.keystroke = 'right ctrl'
                        self.subsequent_sell()
                elif self.keystroke == 'right shift':
                    self.last_keystroke = self.keystroke
                    self.first_buy()
                    self.subsequent_buy()
                elif self.keystroke == 'right ctrl':
                    self.last_keystroke = self.keystroke
                    self.first_sell()
                    self.subsequent_sell()
                elif self.keystroke == 'space':
                    self.last_keystroke = self.keystroke
                    self.first_buy_and_sell()
                    self.subsequent_buy_and_sell()
                elif self.keystroke == 'esc':
                    self.utils.speak('Escape')
                    return
                else:
                    self.last_keystroke = self.keystroke
                    continue

    def run(self):
        try:
            # self.keystroke_thread()
            # return
            self.initial_setup()
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
    if len(sys.argv) < 2:
        print('Usage: python mokhtiyar.py <strategy> <colors_to_check>')
        print('strategy options: "buy&sell", "buy", "sell"')
        print('colors_to_check options: Any combo of: "mb", "st", "lt", "bg"')
        print('Example: py mokhtiyar.py "buy&sell" "mb", "st", "lt"')
        sys.exit(1)

    strategy = sys.argv[1] # buy&sell, buy, sell
    colors_to_check = sys.argv[2:] # All arguments from argv[2] onwards

    if strategy not in ['buy&sell', 'buy', 'sell'] or any(line not in ['mb', 'st', 'lt', 'bg'] for line in colors_to_check):
        print('Invalid strategy or colors_to_check argument.')
        print('Usage: python mokhtiyar.py <strategy> <colors_to_check>')
        print('strategy options: "buy&sell", "buy", "sell"')
        print('colors_to_check options: Any combo of: "mb", "st", "lt", "bg"')
        print('Example: py mokhtiyar.py "buy&sell" "mb", "st", "lt"')
        sys.exit(1)

    trader = PIXEL_TRADER(strategy, colors_to_check)
    trader.run()