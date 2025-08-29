import pyautogui
import time
import os

time.sleep(2)
region = (0,50,1000,760)

os.makedirs("label_imgs", exist_ok=True)

start_time = time.time()
end_time = 0
counter = 0

while end_time < 2400:
    counter += 1
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(f"label_imgs/screenshot_{counter}.png")
    end_time = time.time() - start_time
    print(f"screenshot {counter} taken!")
    time.sleep(0.25)