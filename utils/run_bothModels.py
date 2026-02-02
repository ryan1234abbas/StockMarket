from ultralytics import YOLO
import os
import cv2

'''
run both functional models to create new set of images and manually assess them
'''

old = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt')
yellow = YOLO("/Users/ryanabbas/Desktop/work/StockMarket/runs/detect/combined_model2/weights/best.pt")

folders = ["images/train", "images/val"]
out = "new_training_images"
os.makedirs(out, exist_ok=True)

for folder in folders:
    for img_name in os.listdir(folder):
        if not img_name.endswith((".png", ".jpg")):
            continue
        path = os.path.join(folder, img_name)

        r1 = old(path)
        r2 = yellow(path)

        im1 = r1[0].plot()  # old model preds
        im2 = r2[0].plot()  # yellow model preds

        # Overlay by averaging (or just show both side by side)
        combined = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)

        cv2.imwrite(os.path.join(out, img_name), combined)
