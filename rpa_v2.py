import threading
from tkinter import messagebox
import tkinter
import traceback
import pyautogui
from PIL import ImageDraw
import time
import pyttsx3
 
green_band = (174, 255, 50)
green_bg = (161, 187, 137)
purple_bg = (166, 160, 180)
purple_band = (150, 4, 212)

def show_popup(message):
    # Create a Tkinter root window
    root = tkinter.Tk()
    root.withdraw()  # Hide the root window
    # Display a pop-up message box
    messagebox.showinfo("Popup", message)

def speak(text):
    memory_thread = threading.Thread(target=speak_thread, args=[text])
    memory_thread.start()
 
def speak_thread(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
 
def draw_red_circle(image, x, y, radius=10):
    draw = ImageDraw.Draw(image)
    x1 = x - radius
    y1 = y - radius
    x2 = x + radius
    y2 = y + radius
    draw.ellipse((x1, y1, x2, y2), outline='red')
    # image.show()
 
def get_top_right_coordinates(screenshot, color=(174, 255, 50)):
    screenshot_width, screenshot_height = screenshot.size
    for x in range(screenshot_width - 1, -1, -1):
        for y in range(screenshot_height):
            pixel_color = screenshot.getpixel((x, y))
            if pixel_color == color:
                draw_red_circle(screenshot, x, y)
                return [x, y]
    return None

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
    X_MARGIN_LEFT = 0
    Y_TOP_MARGIN = 0
    X_MARGIN_RIGHT = screen_width - 245
    Y_BOTTOM_MARGIN = screen_height - 50
    pyautogui.hotkey('alt', 'tab')
    start_time = time.time()
    current_status = None
    previous_status = None
    while True:
        screenshot = pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN])
        green_band_coordinates = get_top_right_coordinates(screenshot, green_band)
        green_bg_coordinates = get_top_right_coordinates(screenshot, green_bg)
        purple_band_coordinates = get_top_right_coordinates(screenshot, purple_band)
        purple_bg_coordinates = get_top_right_coordinates(screenshot, purple_bg)
        green_band_x, green_band_y = green_band_coordinates
        green_bg_x, green_bg_y = green_bg_coordinates
        purple_band_x, purple_band_y = purple_band_coordinates
        purple_bg_x, purple_bg_y = purple_bg_coordinates
        if green_band_x < purple_band_x and green_bg_x < purple_bg_x:
            current_status = 'red'
            if current_status != previous_status and previous_status != None:
                speak('Sell')
                # sell()
                break
            previous_status = current_status
        elif green_band_x > purple_band_x and green_bg_x > purple_bg_x:
            current_status = 'green'
            if current_status != previous_status and previous_status != None:
                speak('Buy')
                # buy()
                break
            previous_status = current_status
        else:
            speak('No action')

except Exception as e:
    close()
    print('ERROR ::', e)
    print(traceback.format_exc())
    screenshot.save("error.png")
