import pyautogui
import time

class TabScreenshotter:
    def __init__(self, save_dir="AP/testing"):
        self.save_dir = save_dir

    # def non_max_suppression_fast(boxes, overlapThresh):
#     if len(boxes) == 0:
#         return []

#     boxes = np.array(boxes)
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")

#     pick = []

#     x1 = boxes[:,0]
#     y1 = boxes[:,1]
#     x2 = boxes[:,2]
#     y2 = boxes[:,3]

#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = np.argsort(y2)  # sort by bottom-right y

#     while len(idxs) > 0:
#         last = idxs[-1]
#         i = idxs[-1]
#         pick.append(i)

#         xx1 = np.maximum(x1[i], x1[idxs[:-1]])
#         yy1 = np.maximum(y1[i], y1[idxs[:-1]])
#         xx2 = np.minimum(x2[i], x2[idxs[:-1]])
#         yy2 = np.minimum(y2[i], y2[idxs[:-1]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         overlap = (w * h) / area[idxs[:-1]]

#         i = idxs[-1]
#         suppress = [len(idxs) - 1]  # position of the last element in idxs
#         # Add positions of overlapping boxes
#         suppress.extend(np.where(overlap > overlapThresh)[0])
#         # Remove those positions from idxs
#         idxs = np.delete(idxs, suppress)

#     return boxes[pick].astype("int")

# def black_candles_detect(screenshot_path, template_path, threshold=0.21, margin=100):

#     screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
#     template = cv2.imread(template_path, cv2.IMREAD_COLOR)

#     if screenshot is None or template is None:
#         print("Error loading images")
#         return False

#     screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
#     template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#     boxes = []

#     for scale in np.linspace(0.8, 1.2, 10):  # scale range
#         resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#         th, tw = resized_template.shape[:2]

#         if th > screenshot_gray.shape[0] or tw > screenshot_gray.shape[1]:
#             continue

#         result = cv2.matchTemplate(screenshot_gray, resized_template, cv2.TM_CCOEFF_NORMED)
#         loc = np.where(result >= threshold)
#         for pt in zip(*loc[::-1]):
#             x1 = pt[0]
#             y1 = pt[1]
#             x2 = x1 + tw
#             y2 = y1 + th
#             boxes.append([x1, y1, x2, y2])

#     if not boxes:
#         return False

#     # Apply non-maximum suppression
#     boxes_nms = non_max_suppression_fast(boxes, overlapThresh=0.3)

#     for (x1, y1, x2, y2) in boxes_nms:
#         top_left_shrunk = (x1 + margin, y1 + margin)
#         bottom_right_shrunk = (x2 - margin, y2 - margin)
#         cv2.rectangle(screenshot, top_left_shrunk, bottom_right_shrunk, (0, 0, 255), 2)

#     cv2.imwrite(screenshot_path, screenshot)
#     return True

    def capture_screenshot(self, is_1510):
        screen_width, screen_height = pyautogui.size()
        label = "1510" if is_1510 else "3020"

        if is_1510:
            x = (screen_width // 2) - 150
            width = screen_width - x - 345
        else:
            x = 0
            width = (screen_width // 2) - 130

        y = 150
        height = screen_height - 280

        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot.save(f"{self.save_dir}/{label}.png")
        print(f"Screenshot saved: {self.save_dir}/{label}.png")

    def run(self):
        #Pass in True for 1510 screenshot, and False for 3020        
        self.capture_screenshot(False)  # 3020
        self.capture_screenshot(True)   # 1510

if __name__ == "__main__":
    tab_screenshotter = TabScreenshotter()
    tab_screenshotter.run()
