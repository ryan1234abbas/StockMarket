import time
import cv2
import os
import numpy as np
import pyautogui
import utils
from PIL import Image
import pytesseract
from PIL import Image
import numpy as np
from testing.take_screenshot import TabScreenshotter

class Pandu:
    def __init__(self) -> None:
        self.utils = utils.Utils()
        self.EDGE_DELTA = 5
        self.WIDTH = 20
        self.tab_screenshotter = TabScreenshotter()

    def classify(self, img):

        # Resize to model's expected input size — adjust as needed
        img_resized = img.resize((40, 669))  # (width, height)
        img_resized = img_resized.convert("RGB")  # ensure 3 channels

        img_array = np.array(img_resized) / 255.0
        img_array = img_array[np.newaxis, ...]

        prediction = self.utils.model.predict(img_array, verbose=0)
        prediction = np.argmax(prediction)

        classes = ['DB', 'DT', 'HH', 'HL', 'LH', 'LL']
        return classes[prediction]


    def initial_setup(self):
        self.tab_screenshotter.run()
        #load both saved screenshots
        img_3020 = cv2.imread(f"{self.tab_screenshotter.save_dir}/3020.png")
        img_1510 = cv2.imread(f"{self.tab_screenshotter.save_dir}/1510.png")

        self.top_pixel_3020, self.bottom_pixel_3020 = self.utils.get_right_edge(img_3020)
        self.top_pixel_1510, self.bottom_pixel_1510 = self.utils.get_right_edge(img_1510)

        # Convert to int lists
        self.top_pixel_3020 = list(map(int, self.top_pixel_3020))
        self.bottom_pixel_3020 = list(map(int, self.bottom_pixel_3020))
        self.top_pixel_1510 = list(map(int, self.top_pixel_1510))
        self.bottom_pixel_1510 = list(map(int, self.bottom_pixel_1510))


    def run(self):
        self.initial_setup()
        self.utils.STATUS_3020 = None
        self.utils.STATUS_1510 = None

        region_3020 = self.tab_screenshotter.get_region("3020")
        region_1510 = self.tab_screenshotter.get_region("1510")

        screenshot_3020 = pyautogui.screenshot(region=region_3020).convert("RGB")
        screenshot_1510 = pyautogui.screenshot(region=region_1510).convert("RGB")

        self.utils.PREVIOUS_CLASS_3020 = self.classify(screenshot_3020)
        self.utils.PREVIOUS_CLASS_1510 = self.classify(screenshot_1510)

        while True:
            # ---- Process 3020 ----
            screenshot_3020 = pyautogui.screenshot(region=region_3020)
            # screenshot_3020.save("testing/3020.png")
            # array_3020 = np.array(screenshot_3020.convert("RGB"))

            array_3020 = np.array(screenshot_3020.convert("RGB"))
            array_3020 = self.detect_and_draw_black_candles(array_3020)  # 🔴 draw rectangles
            cv2.imwrite("testing/3020.png", cv2.cvtColor(array_3020, cv2.COLOR_RGB2BGR))  # save with rectangles

            up_pin_3020 = self.utils.get_top_right(array_3020, self.utils.up_pin_point)
            down_pin_3020 = self.utils.get_top_right(array_3020, self.utils.down_pin_point)

            current_class_3020 = self.classify(screenshot_3020)

            self.utils.STATUS_3020 = self._process_logic(
                up_pin_3020,
                down_pin_3020,
                self.utils.PREVIOUS_CLASS_3020,
                current_class_3020,
                tab_name="3020",
                status=self.utils.STATUS_3020
            )

            self.utils.PREVIOUS_CLASS_3020 = current_class_3020

            # ---- Process 1510 ----
            screenshot_1510 = pyautogui.screenshot(region=region_1510)
            # screenshot_1510.save("testing/1510.png")
            # array_1510 = np.array(screenshot_1510.convert("RGB"))

            array_1510 = np.array(screenshot_1510.convert("RGB"))
            array_1510 = self.detect_and_draw_black_candles(array_1510)  # 🔴 draw rectangles
            cv2.imwrite("testing/1510.png", cv2.cvtColor(array_1510, cv2.COLOR_RGB2BGR))  # save with rectangles

            up_pin_1510 = self.utils.get_top_right(array_1510, self.utils.up_pin_point)
            down_pin_1510 = self.utils.get_top_right(array_1510, self.utils.down_pin_point)

            current_class_1510 = self.classify(screenshot_1510)

            self.utils.STATUS_1510 = self._process_logic(
                up_pin_1510,
                down_pin_1510,
                self.utils.PREVIOUS_CLASS_1510,
                current_class_1510,
                tab_name="1510",
                status=self.utils.STATUS_1510
            )

            self.utils.PREVIOUS_CLASS_1510 = current_class_1510


    def _process_logic(self, up_pin_pt, down_pin_pt, previous_class, current_class, tab_name, status):
        if up_pin_pt[1] > down_pin_pt[1] and status == self.utils.PURPLE_STATE:
            self.utils.CURRENT_PIN = self.utils.GREEN_STATE
            if (current_class == 'HL' and previous_class in ['HH', 'LH']):
                self.utils.buy()
                print(f"\n------ [{tab_name}] BUY ------")
                print(f"Prev Class: {previous_class}, Current Class: {current_class}")
                print(up_pin_pt, down_pin_pt)
                print("------------------------------\n")
                return self.utils.GREEN_STATE
        elif up_pin_pt[1] < down_pin_pt[1] and status == self.utils.GREEN_STATE:
            self.utils.CURRENT_PIN = self.utils.PURPLE_STATE
            if (current_class == 'LH' and previous_class in ['LL', 'HL']):
                self.utils.sell()
                print(f"\n------ [{tab_name}] SELL ------")
                print(f"Prev Class: {previous_class}, Current Class: {current_class}")
                print(up_pin_pt, down_pin_pt)
                print("------------------------------\n")
                return self.utils.PURPLE_STATE
        elif up_pin_pt[1] > down_pin_pt[1] and status is None:
            print(f"\n------ [{tab_name}] SELL ------")
            print(f"Current Class: {current_class}")
            print(up_pin_pt, down_pin_pt)
            print("------------------------------\n")
            return self.utils.GREEN_STATE
        elif up_pin_pt[1] < down_pin_pt[1] and status is None:
            print(f"\n------ [{tab_name}] SELL ------")
            print(f"Current Class: {current_class}")
            print(up_pin_pt, down_pin_pt)
            print("------------------------------\n")
            return self.utils.PURPLE_STATE

        return status  # unchanged if no condition matched
    
    def detect_and_draw_black_candles(self, array_rgb, hex_value="#000000", tolerance=30):

        # Convert hex to BGR
        hex_value = hex_value.lstrip("#")
        r, g, b = int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16)
        lower = np.clip([b - tolerance, g - tolerance, r - tolerance], 0, 255)
        upper = np.clip([b + tolerance, g + tolerance, r + tolerance], 0, 255)

        array_bgr = cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR)
        mask = cv2.inRange(array_bgr, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 5 and h > 5:
                cv2.rectangle(array_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    pandu = Pandu()
    pandu.run()



            # def initial_setup(self):        
    #     screen_width, screen_height = pyautogui.size()
    #     X_TOP = 0 #screen_width - 445
    #     Y_TOP = 0
    #     WIDTH = screen_width - 245
    #     HEIGHT = screen_height - 50
    #     pyautogui.hotkey('alt', 'tab')
    #     screenshot_array = np.array(pyautogui.screenshot(region=[X_TOP, Y_TOP, WIDTH, HEIGHT]))
    #     self.top_pixel, self.bottom_pixel = self.utils.get_right_edge(screenshot_array)
    #     self.top_pixel = list(map(int, self.top_pixel))
    #     self.bottom_pixel = list(map(int, self.bottom_pixel))

    # def ocr(self, image):
    #     os.environ['TESSDATA_PREFIX'] = 'C:\\Program Files\\Tesseract-OCR\\tessdata'
    #     original_image = image
        
    #     # Path to Tesseract executable
    #     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    #     # # Read image using OpenCV
    #     # image = cv2.imread('image1.png')
    #     image = np.array(image)

    #     # image_black_to_blue = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0), 255, 0)
    #     # Convert the image to grayscale
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    #     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # # Apply thresholding to preprocess the image
    #     # _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #     # image = screenshot.convert('L')
    #     # image = image.point(lambda x: 0 if x < 128 else 255, '1')
    #     im = Image.fromarray(gray_image)
    #     im.save('threshold_image.png')
    #     # text = pytesseract.image_to_data(image, 'eng', config='--psm 6 --oem 1', output_type=pytesseract.Output.DICT)

    #     text = pytesseract.image_to_string(gray_image, 'eng', config='--psm 6 --oem 1')
    #     print(text)

    #     # # Keras OCR
    #     # print('Keras OCR')
    #     # # os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0
    #     # # `=0
    #     # pipeline = keras_ocr.pipeline.Pipeline()
    #     # prediction_groups = pipeline.recognize([original_image], return_plot=True)
    #     # import pandas as pd
    #     # pred = pd.DataFrame(prediction_groups)
    #     # print(pred)
    #     return text

    # def run(self):
    #     self.initial_setup()
    #     self.previous_up_pin_pt = [-1, -1]
    #     self.previous_down_pin_pt = [-1, -1]
    #     ocr_image = pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH + 5, self.top_pixel[0], 2*self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0]))
    #     self.utils.PREVIOUS_CLASS = self.classify(ocr_image)
    #     while True:
    #         ocr_image = pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH + 5, self.top_pixel[0], 2*self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0]))
    #         screenshot_array = np.array(ocr_image)
    #         up_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.up_pin_point)
    #         down_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.down_pin_point)
    #         self.utils.CURRENT_CLASS = self.classify(ocr_image)
    #         if up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS == self.utils.PURPLE_STATE:
    #             self.utils.CURRENT_PIN = self.utils.GREEN_STATE
    #             if (self.utils.CURRENT_CLASS == 'HL' and self.utils.PREVIOUS_CLASS == 'HH') or (self.utils.CURRENT_CLASS == 'HL' and self.utils.PREVIOUS_CLASS == 'LH'):
    #                 self.utils.buy()
    #                 print()
    #                 print('----------------------------------------------------------------------------')
    #                 print('BUY')
    #                 print('Prev Class', self.utils.PREVIOUS_CLASS)
    #                 print('Current class', self.utils.CURRENT_CLASS)
    #                 print(up_pin_pt, down_pin_pt)
    #                 print('----------------------------------------------------------------------------')
    #                 print()
    #                 self.utils.STATUS = self.utils.GREEN_STATE
    #         elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS == self.utils.GREEN_STATE:
    #             self.utils.CURRENT_PIN = self.utils.PURPLE_STATE
    #             if (self.utils.CURRENT_CLASS == 'LH' and self.utils.PREVIOUS_CLASS == 'LL') or (self.utils.CURRENT_CLASS == 'LH' and self.utils.PREVIOUS_CLASS == 'HL'):
    #                 self.utils.sell()
    #                 print()
    #                 print('----------------------------------------------------------------------------')
    #                 print('SELL')
    #                 # print('Prev Class', self.utils.PREVIOUS_CLASS)
    #                 print('Current class', self.utils.CURRENT_CLASS)
    #                 print(up_pin_pt, down_pin_pt)
    #                 print('----------------------------------------------------------------------------')
    #                 print()
    #                 self.utils.STATUS = self.utils.PURPLE_STATE
    #         elif up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS is None:
    #             # self.utils.buy()
    #             print()
    #             print('----------------------------------------------------------------------------')
    #             print('SELL')
    #             # print('Prev Class', self.utils.PREVIOUS_CLASS)
    #             print('Current class', self.utils.CURRENT_CLASS)
    #             print(up_pin_pt, down_pin_pt)
    #             print('----------------------------------------------------------------------------')
    #             print()
    #             self.utils.STATUS = self.utils.GREEN_STATE
    #         elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS is None:
    #             # self.utils.sell()
    #             print()
    #             print('----------------------------------------------------------------------------')
    #             print('SELL')
    #             # print('Prev Class', self.utils.PREVIOUS_CLASS)
    #             print('Current class', self.utils.CURRENT_CLASS)
    #             print(up_pin_pt, down_pin_pt)
    #             print('----------------------------------------------------------------------------')
    #             print()
    #             self.utils.STATUS = self.utils.PURPLE_STATE

            # if self.utils.CURRENT_CLASS != self.utils.PREVIOUS_CLASS:
            #     # print('Prev Class', self.utils.PREVIOUS_CLASS)
            #     # print('Current class', self.utils.CURRENT_CLASS)
            #     # print(up_pin_pt, down_pin_pt)
            #     if (self.utils.CURRENT_CLASS == 'HL' and self.utils.PREVIOUS_CLASS == 'HH') or (self.utils.CURRENT_CLASS == 'HL' and self.utils.PREVIOUS_CLASS == 'LH'): # and self.utils.STATUS == self.utils.PURPLE_STATE
            #         while True:
            #             screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH, self.top_pixel[0], self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0])))
            #             up_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.up_pin_point)
            #             down_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.down_pin_point)
            #             if up_pin_pt[1] > down_pin_pt[1]:
            #                 # self.utils.speak('BUY')
            #                 # self.utils.close()
            #                 self.utils.buy()
            #                 # self.utils.reverse()
            #                 print()
            #                 print('----------------------------------------------------------------------------')
            #                 print('BUY')
            #                 print('Prev Class', self.utils.PREVIOUS_CLASS)
            #                 print('Current class', self.utils.CURRENT_CLASS)
            #                 print(up_pin_pt, down_pin_pt)
            #                 # text =self.utils.PREVIOUS_CLASS + self.utils.CURRENT_CLASS
            #                 # self.utils.speak(text)
            #                 print('----------------------------------------------------------------------------')
            #                 print()
            #                 self.utils.STATUS = self.utils.GREEN_STATE
            #                 break
            #     elif (self.utils.CURRENT_CLASS == 'LH' and self.utils.PREVIOUS_CLASS == 'LL') or (self.utils.CURRENT_CLASS == 'LH' and self.utils.PREVIOUS_CLASS == 'HL'): # and self.utils.STATUS == self.utils.GREEN_STATE
            #         while True:
            #             screenshot_array = np.array(pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH, self.top_pixel[0], self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0])))
            #             up_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.up_pin_point)
            #             down_pin_pt = self.utils.get_top_right(screenshot_array, self.utils.down_pin_point)
            #             if up_pin_pt[1] < down_pin_pt[1]:
            #                 # self.utils.speak('SELL')
            #                 # self.utils.close()
            #                 self.utils.sell()
            #                 # self.utils.reverse()
            #                 print()
            #                 print('----------------------------------------------------------------------------')
            #                 print('SELL')
            #                 print('Prev Class', self.utils.PREVIOUS_CLASS)
            #                 print('Current class', self.utils.CURRENT_CLASS)
            #                 print(up_pin_pt, down_pin_pt)
            #                 print('----------------------------------------------------------------------------')
            #                 print()
            #                 self.utils.STATUS = self.utils.PURPLE_STATE
            #                 break
            # #elif up_pin_pt[1] > down_pin_pt[1] and self.utils.STATUS == self.utils.PURPLE_STATE:
            #     # self.utils.speak('Green')
            #     # self.utils.close()
            #     # self.utils.STATUS = None
            # # elif up_pin_pt[1] < down_pin_pt[1] and self.utils.STATUS == self.utils.GREEN_STATE:
            #     # self.utils.speak('Cyan')
            #     # self.utils.close()
            #     # self.utils.STATUS = None
            # self.utils.PREVIOUS_CLASS = self.utils.CURRENT_CLASS