import pyautogui
import time
import os

time.sleep(5)
region = (0,50,1000,610)

os.makedirs("label_candle_imgs", exist_ok=True)

start_time = time.time()
end_time = 0
counter = 134

while end_time < 2400:
    counter += 1
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(f"label_candle_imgs/screenshot_{counter}.png")
    end_time = time.time() - start_time
    print(f"screenshot {counter} taken!")
    time.sleep(2)