from ultralytics import YOLO

def test_model():
    # Load your trained model
    model = YOLO("/Users/ryanabbas/Desktop/work/StockMarket/runs/detect2/train7/weights/last.pt")

    # Run inference on an image
    results = model('/Users/ryanabbas/Desktop/work/StockMarket/label_imgs_xml/screenshot_151.png')

    results[0].show()

