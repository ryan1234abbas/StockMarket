import pyautogui
import time
import os

region = (0,50,1000,610)

os.makedirs("all_images", exist_ok=True)
counter = 216

while True: 
    time.sleep(2)
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(f"all_images/screenshot_{counter}.png")
    counter+=1
    print(f"screenshot taken! {counter}")

#213