from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("/Users/ryanabbas/Desktop/work/StockMarket/runs/detect/combined_model2/weights/best.pt")

# Test on your image
image_path = "images/train/screenshot_354.png"
results = model(image_path, conf=0.1)

# Check all detected classes
print("Detected classes:")
if results[0].boxes is not None:
    class_counts = {}
    for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
        class_name = model.names[int(cls)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"  {class_name}: {float(conf):.3f}")
    
    print(f"\nSummary:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")
else:
    print("No detections")

# Visualize with OpenCV
annotated = results[0].plot()  # Get image with bounding boxes
cv2.imshow("YOLO Predictions", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the visualization
cv2.imwrite("predictions_result.jpg", annotated)
print(f"\nSaved visualization to: predictions_result.jpg")