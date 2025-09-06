import pyautogui
import time

time.sleep(3)  # gives you 3 seconds to open the screen where the button is

buybtn = pyautogui.locateCenterOnScreen('buy_sell/buy2.png')
sellbtn = pyautogui.locateCenterOnScreen('buy_sell/sell2.png')

print("Button found at:", buybtn)
print("Button found at:", sellbtn)