import threading
from tkinter import messagebox
import tkinter
import pyautogui
from PIL import Image, ImageGrab, ImageDraw
import time
import numpy as np
import pyttsx3
 
green_yellow_band = (174, 255, 50)
green_bg = (137, 162, 137)
pink_bg = (187, 137, 137)
pink_band = (255, 4, 255)

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
 
def mask_color(array1, array2, color):
    mask1 = np.all(array1 == color, axis=-1)
    mask2 = np.all(array2 == color, axis=-1)
    array1[mask1] = [0, 0, 0]
    array2[mask1] = [0, 0, 0]
    array1[mask2] = [0, 0, 0]
    array2[mask2] = [0, 0, 0]
    return array1, array2
 
try:
    screen_width, screen_height = pyautogui.size()
    X_MARGIN_LEFT = 0
    Y_TOP_MARGIN = 0
    X_MARGIN_RIGHT = screen_width - 245
    Y_BOTTOM_MARGIN = screen_height - 50
    pyautogui.hotkey('alt', 'tab')
    start_time = time.time()
    screenshot = pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN])
    get_top_right_coordinates(screenshot, pink_band)
    screenshot.save("screenshot.png")

except Exception as e:
    close()
    print('ERROR ::', e)
    # screenshot.save("error.png")
