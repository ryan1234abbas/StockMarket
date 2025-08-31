from ultralytics import YOLO

def test_model():
    # Load your trained model
    model = YOLO("/Users/ryanabbas/Desktop/work/StockMarket/runs/detect2/train8/weights/best.pt")

    # Run inference on an image
    results = model('/Users/ryanabbas/Desktop/work/StockMarket/images/train/screenshot_109.png')

    results[0].show()

test_model()