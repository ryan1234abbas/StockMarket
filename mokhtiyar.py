# Safe exit of threads: press esc. do not press ctrl + C or ctrl + X
# Keystroke will perform shortcuts defined in the active window along with the actions defined in the script.

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

    def __init__(self):
        self.utils = Utils()

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

    def keystroke_thread(self):
        while True:
            self.keystroke = keyboard.read_key()  # This will block until a key is pressed
            # self.utils.speak(self.keystroke)
            # self.last_keystroke = self.keystroke
            if self.keystroke == 'esc':
                sys.exit()

    def keystroke_listener(self):
        self.last_keystroke = 'right'
        self.keystroke = None
        self.key_thread = threading.Thread(target=self.keystroke_thread)
        self.key_thread.start()
        while True:
            time.sleep(0.01)
            if self.keystroke:
                print(self.last_keystroke, self.keystroke)
                if self.keystroke == 'enter':
                    print('Closed by user')
                #     # self.utils.close()
                #     if self.last_keystroke == 'right':
                #         self.utils.speak('buy or sell')
                #         # self.run_buy_or_sell()
                    self.last_keystroke = self.keystroke
                    self.keystroke = None
                #     # break
                elif self.keystroke == 'up':
                #     # self.utils.speak('Up')
                    # self.utils.speak('run_buy')
                #     # self.run_buy()
                    self.last_keystroke = self.keystroke
                    self.keystroke = None
                #     # break
                elif self.keystroke == 'down':
                #     # self.utils.speak('Down')
                    # self.utils.speak('run_sell')
                #     # self.run_sell()
                    self.last_keystroke = self.keystroke
                    self.keystroke = None
                #     # break
                elif self.keystroke == 'right':
                #     # self.utils.speak('Right')
                    # self.utils.speak('buy or sell')
                #     # self.run_buy_or_sell()
                    self.last_keystroke = self.keystroke
                    self.keystroke = None
                #     # break
                elif self.keystroke == 'esc':
                    self.utils.speak('Escape')
                    return
                else:
                    self.last_keystroke = self.keystroke
                    self.keystroke = None
                    continue

    def keystroke_listener_TBD(self):
        self.last_keystroke = 'a'
        self.key_thread = threading.Thread(target=self.utils.get_coordinates)
        # self.stop_events = []
        # stop_event = threading.Event()
        # self.utils.buy_thread = threading.Thread(target=self.run_buy_or_sell)
        # self.utils.buy_thread.start()
        # self.stop_events.append(stop_event)
        while True:
            self.keystroke = keyboard.read_key()  # This will block until a key is pressed
            print(f'Key pressed: {keystroke}')
            if self.keystroke == 'enter':
                self.utils.speak('Enter')
                # print('Closed by user')
                # self.utils.close()
                # if self.last_keystroke == 'right':
                #     self.handle_thread(self.run_buy_or_sell)
                # break
            elif self.keystroke == 'up':
                self.utils.speak('Up')
                # self.handle_thread(self.run_buy)
                # break
            elif self.keystroke == 'down':
                self.utils.speak('Down')
                # self.handle_thread(self.run_sell)
                # break
            elif self.keystroke == 'right':
                self.utils.speak('Right')
                # self.handle_thread(self.run_buy_or_sell)
                # break
            elif self.keystroke == 'esc':
                self.utils.speak('Escape')
                break
            self.last_keystroke = self.keystroke

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