import threading
from tkinter import messagebox
import tkinter
import pyautogui
from PIL import Image, ImageGrab, ImageDraw
import time
import numpy as np
import datetime
import pyttsx3
from speedtest import Speedtest
import psutil
from memory_profiler import profile
 

previous_screenshot_array = None
previous_screenshot = None
screenshot_array = None
count=0

# X - left to right -> 0 to 999
# Y - top to down -> 0 to 999
# 8.6 - pixels b/w 2 candles

BUY = -1
X_MARGIN_RIGHT = 1545
X_MARGIN_LEFT = 1469
Y_TOP_MARGIN = 53
Y_BOTTOM_MARGIN = 1031
green_top_coordinates = []
gx = None
gy = None
rx = None
ry = None
red_bottom_coordinates = []

 
def log_network():
    speedtest = Speedtest()
    download_speed = speedtest.download() / 1024 / 1024  # Convert Mbps to MB/s
    upload_speed = speedtest.upload() / 1024 / 1024  # Convert Mbps to MB/s
    print(f"Download speed: {download_speed:.2f} MB/s")
    print(f"Upload speed: {upload_speed:.2f} MB/s")
 
def log_metrics():
    """Log memory usage."""
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
    print(f"Memory Usage: {memory_usage:.2f} GB")
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    disk_usage = psutil.disk_usage('/').percent
    print(f"Memory: {memory_usage:.2f}%, CPU: {cpu_usage:.2f}%, Disk: {disk_usage:.2f}%")

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
 
def green_bottom(screenshot):
    global gx
    global gy
    for x in range(X_MARGIN_RIGHT, -1, -1):
        for y in range(Y_BOTTOM_MARGIN, -1, -1):
            pixel_color = screenshot.getpixel((x, y))
            if pixel_color == (53, 206, 53):
                # green_bottom_coordinates = [x,y]
                gx = x
                gy = y
                # draw_red_circle(screenshot, x, y)
                return
 
def green_top(screenshot):
    global green_top_coordinates
    for x in range(X_MARGIN_RIGHT, -1, -1):
        for y in range(Y_BOTTOM_MARGIN):
            pixel_color = screenshot.getpixel((x, y))
            if pixel_color == (53, 206, 53):
                green_top_coordinates = [x,y]
                return
 
def red_bottom(screenshot):
    global red_bottom_coordinates
    for x in range(X_MARGIN_RIGHT, -1, -1):
        for y in range(Y_BOTTOM_MARGIN, -1, -1):
            pixel_color = screenshot.getpixel((x, y))
            if pixel_color == (255, 4, 4):
                red_bottom_coordinates =[x, y]
                return
 
def red_top(screenshot):
    global rx
    global ry
    for x in range(X_MARGIN_RIGHT, -1, -1):
        for y in range(Y_TOP_MARGIN, Y_BOTTOM_MARGIN):
            pixel_color = screenshot.getpixel((x, y))
            if pixel_color == (255, 4, 4):
                # red_top_coordinates = [x, y]
                rx = x
                ry = y
                # draw_red_circle(screenshot, x, y)
                return
 
def optimal_red_top(screenshot):
    global rx, ry
    screenshot_array = np.array(screenshot)
 
    # Create a mask for red pixels (255, 4, 4)
    red_mask = np.all(screenshot_array == [255, 4, 4], axis=-1)
    red_indices = np.argwhere(red_mask)
    max_rx_index = np.argmax(red_indices[:, 1])
 
    ry, rx = red_indices[max_rx_index]
 
def optimal_green_bottom(screenshot):
    global gx, gy
    screenshot_array = np.array(screenshot)
 
    # Create a mask for green pixels (53, 206, 53)
    green_mask = np.all(screenshot_array == [53, 206, 53], axis=-1)
    green_indices = np.argwhere(green_mask)
    max_gx_index = np.argmax(green_indices[:, 1])
 
    furthest_right_indices = green_indices[green_indices[:, 1] == green_indices[max_gx_index, 1]]
    gy, gx = furthest_right_indices[-1]
 
def get_coordinates():
    point = pyautogui.position()
    print(point)
 
def buy():
    pyautogui.click(1725, 55)
 
def sell():
    pyautogui.click(1855, 48)
 
def reverse():
    pyautogui.click(x=1732, y=168)
 
def close():
    pyautogui.click(x=1859, y=168)
 
def mask_color(array1, array2, color):
    mask1 = np.all(array1 == color, axis=-1)
    mask2 = np.all(array2 == color, axis=-1)
    array1[mask1] = [0, 0, 0]
    array2[mask1] = [0, 0, 0]
    array1[mask2] = [0, 0, 0]
    array2[mask2] = [0, 0, 0]
    return array1, array2
 
def same_line_transaction():
    global previous_screenshot_array
    global screenshot_array
    previous_screenshot_array, screenshot_array = mask_color(previous_screenshot_array, screenshot_array, [223, 185, 137]) # Buy/Sell Line
    previous_screenshot_array, screenshot_array = mask_color(previous_screenshot_array, screenshot_array, [219, 166, 35]) # Yellow line
 
    if np.array_equal(previous_screenshot_array, screenshot_array):
        # speak('Same')
        BUY = -1
        return True
    else:
        # speak('Not Same')
        return False
 
try:
    screen_width, screen_height = pyautogui.size()
    X_MARGIN_LEFT = 100
    Y_TOP_MARGIN = 50
    X_MARGIN_RIGHT = screen_width - 500
    Y_BOTTOM_MARGIN = screen_height - 150
    pyautogui.hotkey('alt', 'tab')
    start_time = time.time()
 
    while(1):
        # screenshot = pyautogui.screenshot()
        screenshot = pyautogui.screenshot(region=[X_MARGIN_LEFT, Y_TOP_MARGIN, X_MARGIN_RIGHT, Y_BOTTOM_MARGIN])
        # get_coordinates()
        # draw_red_circle(screenshot,x=X_MARGIN_RIGHT, y=Y_TOP_MARGIN)
        # draw_red_circle(screenshot,x=X_MARGIN_RIGHT, y=Y_BOTTOM_MARGIN)
 
        threads = []
        thread = threading.Thread(target=optimal_green_bottom, args=([screenshot]))
        threads.append(thread)
        thread.start()
        thread = threading.Thread(target=optimal_red_top, args=([screenshot]))
        threads.append(thread)
        thread.start()
 
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
 
        # Buy
        if BUY == 0:
            if  gx > rx:
                cropped_image = screenshot.crop((gx - 25, Y_TOP_MARGIN, gx - 4, Y_BOTTOM_MARGIN))
                screenshot_array = np.array(cropped_image)
                cropped_image = previous_screenshot.crop((gx - 25, Y_TOP_MARGIN, gx - 4, Y_BOTTOM_MARGIN))
                previous_screenshot_array = np.array(cropped_image)
                if not same_line_transaction():
                    reverse()
                else:
                    continue
                # reverse()
                BUY = 1
        # Sell
        elif BUY == 1:
            if gx < rx:
                cropped_image = screenshot.crop((rx - 25, Y_TOP_MARGIN, rx - 4, Y_BOTTOM_MARGIN))
                screenshot_array = np.array(cropped_image)
                cropped_image = previous_screenshot.crop((rx - 25, Y_TOP_MARGIN, rx - 4, Y_BOTTOM_MARGIN))
                previous_screenshot_array = np.array(cropped_image)
                if not same_line_transaction():
                    reverse()
                else:
                    continue
                # reverse()
                BUY = 0
                count = count + 1
                # Change count to increase or decrease the number of transactions
                if count == 6:
                    close()
                    break
        # 1st Buy
        elif BUY == -1 and gx > rx:
                cropped_image = screenshot.crop((gx - 25, Y_TOP_MARGIN, gx - 4, Y_BOTTOM_MARGIN))
                screenshot_array = np.array(cropped_image)
                buy()
                BUY = 1
                count = 0
        # 1st Sell
        elif BUY == -1 and  gx < rx:
                cropped_image = screenshot.crop((rx - 25, Y_TOP_MARGIN, rx - 4, Y_BOTTOM_MARGIN))
                screenshot_array = np.array(cropped_image)
                sell()
                BUY = 0
                count = 0
        previous_screenshot = screenshot
 
    end_time = time.time()
    print(f"Time taken: **{round(end_time - start_time, 2)} seconds**")
except Exception as e:
    close()
    print('ERROR ::', e)
    screenshot.save("error.png")
