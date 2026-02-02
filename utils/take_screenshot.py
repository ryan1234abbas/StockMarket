import pyautogui
import os
import time

class TabScreenshotter:
    def __init__(self):
        self.save_dir = os.path.dirname(os.path.abspath(__file__))

    def get_region(self, tab_name):
        screen_width, screen_height = pyautogui.size()
        if tab_name == "3020":
            x = 0
            width = (screen_width // 2) - 130
        elif tab_name == "1510":
            x = (screen_width // 2) - 150
            width = screen_width - x - 345
        else:
            raise ValueError(f"No region defined for tab {tab_name}")

        y = 150
        height = screen_height - 280

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid region for tab {tab_name}: width={width}, height={height}")

        return (x, y, width, height)

    def get_tab_offset(self, tab_name):
        if tab_name == "3020":
            return 100, 150  # example: left=100px, top=150px
        elif tab_name == "1510":
            return 300, 150
        return 0, 0

    
    def capture_screenshot(self, tab_name):
        region = self.get_region(tab_name)
        screenshot = pyautogui.screenshot(region=region)
        save_path = os.path.join(self.save_dir, f"{tab_name}.png")
        screenshot.save(save_path)
        print(f"Captured and saved screenshot for tab {tab_name} at {save_path}")
        return screenshot


    def run(self):
        self.capture_screenshot("3020")
        self.capture_screenshot("1510")

if __name__ == "__main__":
    tab_screenshotter = TabScreenshotter()
    tab_screenshotter.run()
