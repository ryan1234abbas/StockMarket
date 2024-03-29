import sys
import pyautogui
import utils
import numpy as np

class ParameterizedTrade:

    def __init__(self,strategy, lines):
        self.strategy = strategy
        self.lines = lines
        self.utils = utils.Utils()
        self.run_initial_setup()
        if strategy == 'buy&sell':
            self.run_buy_sell()
        elif strategy == 'buy':
            self.run_buy()
        elif strategy == 'sell':
            self.run_sell()

    def run_initial_setup(self):
        """Get screenshot, find the right edge of the green and purple background, and set the status accordingly."""
        screen_width, screen_height = pyautogui.size()
        X_MARGIN_LEFT = 0 #screen_width - 445
        Y_TOP_MARGIN = 0
        X_MARGIN_RIGHT = screen_width - 245
        Y_BOTTOM_MARGIN = screen_height - 50
        pyautogui.hotkey('alt', 'tab')
        screenshot_array = np.array(pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN]))
        self.top_pixel, self.bottom_pixel = self.utils.get_right_edge(screenshot_array)
        print(self.top_pixel, self.bottom_pixel)

    def run_buy_sell(self):
        print('Running buy and sell')

    def run_buy(self):
        print('Running buy')
    
    def run_sell(self):
        print('Running sell')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python parameterized_trade.py <strategy> <lines>')
        print('strategy options: "buy&sell", "buy", "sell"')
        print('lines options: Any combo of: "mb", "st", "lt", "bg"')
        print('Example: py parameterized_trade.py "buy&sell" "mb", "st", "lt"')
        sys.exit(1)
    
    strategy = sys.argv[1] # buy&sell, buy, sell
    lines = sys.argv[2:] # All arguments from argv[2] onwards

    if strategy not in ['buy&sell', 'buy', 'sell'] or any(line not in ['mb', 'st', 'lt', 'bg'] for line in lines):
        print('Invalid strategy or lines argument.')
        print('Usage: python parameterized_trade.py <strategy> <lines>')
        print('strategy options: "buy&sell", "buy", "sell"')
        print('lines options: Any combo of: "mb", "st", "lt", "bg"')
        print('Example: py parameterized_trade.py "buy&sell" "mb", "st", "lt"')
        sys.exit(1)

    trader = ParameterizedTrade(strategy, lines)
