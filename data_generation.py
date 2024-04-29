import time
import os
import numpy as np
import pyautogui
import utils
import numpy as np

class Pandu:
    def __init__(self) -> None:
        self.utils = utils.Utils()
        self.EDGE_DELTA = 5
        self.WIDTH = 20
        # self.model = tf.keras.models.load_model('mask_model.h5')

    def classify(self, img):
        img = np.array(img)
        img = img / 255.0
        img = img[np.newaxis, ...]

        prediction = self.model.predict(img, verbose=0)
        prediction = np.argmax(prediction)

        # if prediction == 0:
        #     print('DT')
        # elif prediction == 1:
        #     print('HH')
        # elif prediction == 2:
        #     print('HL')
        # elif prediction == 3:
        #     print('LH')
        # else:
        #     print('LL')
        return int(prediction)

    def generate_data(self):
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

        # Generate dataset for training
        count = 5000
        while True:
            count += 1
            if count == 50000:
                break
            ocr_image = pyautogui.screenshot(region=(self.top_pixel[1] - self.WIDTH + 5, self.top_pixel[0], 2*self.WIDTH, self.bottom_pixel[0] - self.top_pixel[0]))
            filename = os.path.join('dataset', str(count) + '.png')
            time.sleep(5)
            ocr_image.save(filename)


if __name__ == '__main__':
    pandu = Pandu()
    pandu.generate_data()
