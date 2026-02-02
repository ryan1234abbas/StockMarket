import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/final_model/weights/best.pt")

results = model("images/val/screenshot_1769985731.46.png")

# Get the image with boxes drawn
img_with_boxes = results[0].plot()

# OpenCV popup
cv2.imshow("Prediction", img_with_boxes)
cv2.waitKey(0)  # wait until you press a key
cv2.destroyAllWindows()
