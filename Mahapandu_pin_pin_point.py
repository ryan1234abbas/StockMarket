import time
import cv2
import os
import numpy as np
import pyautogui
import utils
from PIL import Image
import pytesseract
# import keras_ocr

class Pandu:
    def __init__(self) -> None:
        self.utils = utils.Utils()
        self.EDGE_DELTA = 5
        self.WIDTH = 20

    def initial_setup(self):
        screen_width, screen_height = pyautogui.size()
        X_TOP = 0 #screen_width - 445
        Y_TOP = 0
        WIDTH = screen_width - 245
        HEIGHT = screen_height - 50
        pyautogui.hotkey('alt', 'tab')
        screenshot_array = np.array(pyautogui.screenshot(region=[X_TOP, Y_TOP, WIDTH, HEIGHT]))
        self.top_pixel, self.bottom_pixel = self.utils.get_right_edge(screenshot_array)
        self.top_pixel = list(map(int, self.top_pixel))
        self.bottom_pixel = list(map(int, self.bottom_pixel))
        count = 2500
        while True:
            count += 1
            if count == 50000:
                break
            ocr_image = pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH + 5, self.top_pixel[0], 2*self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0]))
            filename = os.path.join('dataset', str(count) + '.png')
            time.sleep(3)
            ocr_image.save(filename)
        # self.ocr(ocr_image)

    def ocr(self, image):
        os.environ['TESSDATA_PREFIX'] = 'C:\\Program Files\\Tesseract-OCR\\tessdata'
        original_image = image
        
        # Path to Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # # Read image using OpenCV
        # image = cv2.imread('image1.png')
        image = np.array(image)

        # image_black_to_blue = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0), 255, 0)
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Apply thresholding to preprocess the image
        # _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # image = screenshot.convert('L')
        # image = image.point(lambda x: 0 if x < 128 else 255, '1')
        im = Image.fromarray(gray_image)
        im.save('threshold_image.png')
        # text = pytesseract.image_to_data(image, 'eng', config='--psm 6 --oem 1', output_type=pytesseract.Output.DICT)

        text = pytesseract.image_to_string(gray_image, 'eng', config='--psm 6 --oem 1')
        print(text)

        # # Keras OCR
        # print('Keras OCR')
        # # os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0
        # # `=0
        # pipeline = keras_ocr.pipeline.Pipeline()
        # prediction_groups = pipeline.recognize([original_image], return_plot=True)
        # import pandas as pd
        # pred = pd.DataFrame(prediction_groups)
        # print(pred)
        return text

    def run(self):
        self.initial_setup()
        self.previous_up_pin_pt = [-1, -1]
        self.previous_down_pin_pt = [-1, -1]
        self.utils.STATUS = None
        while True:
            screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH, self.top_pixel[0], self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0])))
            up_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.up_pin_point)
            down_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.down_pin_point)
            # screenshot = Image.fromarray(screenshot_array)
            # self.utils.draw_circle(screenshot, int(up_pin_pt[1]), int(up_pin_pt[0]))
            # self.utils.draw_circle(screenshot, int(down_pin_pt[1]), int(down_pin_pt[0]))
            # print(up_pin_pt, down_pin_pt)
            if up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS == self.utils.PURPLE_STATE:
                # self.utils.speak('Green')
                self.utils.close()
                self.utils.buy()
                # self.utils.reverse()
                self.utils.STATUS = self.utils.GREEN_STATE
                # if up_pin_pt[0] > self.previous_up_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_up_pin_pt = up_pin_pt
            elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS == self.utils.GREEN_STATE:
                # self.utils.speak('Cyan')
                self.utils.close()
                self.utils.sell()
                # self.utils.reverse()
                self.utils.STATUS = self.utils.PURPLE_STATE
                # if down_pin_pt[0] > self.previous_down_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_down_pin_pt = down_pin_pt
            elif up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS is None:
                # self.utils.speak('Green')
                self.utils.buy()
                self.utils.STATUS = self.utils.GREEN_STATE
                # if up_pin_pt[0] > self.previous_up_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_up_pin_pt = up_pin_pt
            elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS is None:
                # self.utils.speak('Cyan')
                self.utils.sell()
                self.utils.STATUS = self.utils.PURPLE_STATE
                # if down_pin_pt[0] > self.previous_down_pin_pt[0]:
                #     self.utils.speak('Down')
                previous_down_pin_pt = down_pin_pt


if __name__ == '__main__':
    pandu = Pandu()
    # pandu.run()
    pandu.initial_setup()
    # image = Image.open('temp.png')
    # pandu.ocr(image)
