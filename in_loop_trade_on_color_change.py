# Trading on change of colors in a band
import threading
import traceback
import numpy as np
import pyautogui
from PIL import ImageDraw, Image
import time
import pyttsx3

green_band = [174, 255, 50]
green_bg = [161, 187, 137]
purple_bg = [166, 160, 180]
purple_band = [150, 4, 212]

def speak(text):
    memory_thread = threading.Thread(target=speak_thread, args=[text])
    memory_thread.start()

def speak_thread(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def draw_circle(image, x, y, radius=10):
    draw = ImageDraw.Draw(image)
    x1 = x - radius
    y1 = y - radius
    x2 = x + radius
    y2 = y + radius
    draw.ellipse((x1, y1, x2, y2), outline='red')
    # image.show()

def get_top_right_x(screenshot_array, color):
    mask = np.all(screenshot_array == color, axis=-1)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return -1
    max_index = np.argmax(indices[:, 1])
    return indices[max_index][1]

def get_coordinates():
    point = pyautogui.position()
    print(point)

def buy():
    pyautogui.click(x=1798, y=42)

def sell():
    pyautogui.click(x=1876, y=110)

def reverse():
    pyautogui.click(x=1798, y=145)

def close():
    pyautogui.click(x=1876, y=142)

try:
    screen_width, screen_height = pyautogui.size()
    X_MARGIN_LEFT = 0 #screen_width - 445
    Y_TOP_MARGIN = 0
    X_MARGIN_RIGHT = screen_width - 245
    Y_BOTTOM_MARGIN = screen_height - 50
    pyautogui.hotkey('alt', 'tab')
    start_time = time.time()
    current_status = None
    previous_status = None
    count = 0
    while True:
        screenshot = np.array(pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN]))
        # pixel = get_top_right_x()
        # pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN]Clinical_TWC).save("screenshot.png")
        green_band_x = get_top_right_x(screenshot, green_band)
        purple_band_x = get_top_right_x(screenshot, purple_band)
        if green_band_x < purple_band_x:# and green_bg_x < purple_bg_x:
            current_status = 'red'
            if current_status != previous_status and previous_status != None:
                if count == 0:
                    # speak('Sell')
                    sell()
                else:
                    reverse()
                count += 1
            previous_status = current_status
        elif green_band_x > purple_band_x:# and green_bg_x > purple_bg_x:
            current_status = 'green'
            if current_status != previous_status and previous_status != None:
                if count == 0:
                    # speak('Buy')
                    buy()
                else:
                    reverse()
                count += 1
            previous_status = current_status
        else:
            speak('No action')
            break

except Exception as e:
    close()
    print('ERROR ::', e)
    print(traceback.format_exc())
    Image.fromarray(screenshot).save("error.png")
