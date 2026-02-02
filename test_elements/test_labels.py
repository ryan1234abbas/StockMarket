from ultralytics import YOLO

def test_model():
    # Load your trained model
    model = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt')

    # Run inference on an image
    results = model('/Users/ryanabbas/Desktop/work/StockMarket/images/train/screenshot_175.png')

    results[0].show()

    for box in results[0].boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()   # bounding box corners
        conf = float(box.conf[0])               # confidence
        cls = int(box.cls[0])                   # class id
        label = results[0].names[cls]           # class name

        print(f"Class: {label}, Conf: {conf:.2f}, Coords: ({x0:.1f}, {y0:.1f}), ({x1:.1f}, {y1:.1f})")

test_model()