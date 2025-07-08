from ultralytics import YOLO
import cv2

def execute_model():
    # Load base model (or a previous checkpoint)
    yolo_model = YOLO('yolov8m.pt')

    # Train the model
    yolo_model.train(
    #data='/Users/koshabbas/Desktop/work/stock_market/candle_data.yaml',
    #data='/Users/ryanabbas/Desktop/work/StockMarket/candle_data.yaml',
    data='candle_data.yaml',
    model='yolov8m.pt',
    epochs=100,
    imgsz=1280,
    batch=16,
    name='train_19',
    #device='cpu',  
    device='cuda',
    optimizer='AdamW',
    lr0=0.0005,
    patience=25,
    augment=True,
    scale=0.5,           # Allow zoom-in/out
    translate=0.05,       # Slight vertical/horizontal shift
    perspective=0.0,      # Keep candle shape stable
    shear=0.0,            # Don't shear candles
    flipud=0.0,           # Don't flip upside down
    fliplr=0.5,           # Horizontal flips are okay
    mosaic=1.0,           # Retain variety
    rect=True,            # Keep candle aspect ratio
    cache='disk',         # Faster and deterministic loading
    save=True,
    save_period=10,
    verbose=True,
    workers=4,
    plots=False,
    amp=True,
    )

def asses_performance(model):
    '''Use this to assess model performance'''

    # Image path
    img_path = '/Users/koshabbas/Desktop/work/stock_market/images/train/Screenshot 2025-06-28 at 4.51.48 PM.png'

    # Run prediction with low confidence threshold
    results = model.predict(source=img_path, conf=0.1, iou=0.1, imgsz=1024)  # lower conf, bigger size for better detection

    # Load image with OpenCV for drawing
    img = cv2.imread(img_path)

    for result in results:
        boxes = result.boxes  # Boxes object for current image
        for box in boxes:
            # Get box coordinates (xyxy)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()  # Confidence score

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put confidence text
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show image with detections
    cv2.imshow("Detections with confidence", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hard_coded():
    '''Use this to check hardcoded bounding boxes'''
    import cv2

    # Load image
    img_path = '/Users/koshabbas/Desktop/work/stock_market_imgs/images/train/Screenshot 2025-06-27 at 6.56.18 PM.png'
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Load corresponding label txt
    label_path = '/Users/koshabbas/Desktop/work/stock_market_imgs/labels/train/Screenshot 2025-06-27 at 6.56.18 PM.txt'

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        cls, x_c, y_c, bw, bh = line.strip().split()
        x_c, y_c, bw, bh = map(float, (x_c, y_c, bw, bh))

        # Convert normalized to absolute coords
        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow('img with boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def model_on_vid(model, og_vid):
    # Load model

    # Open input video
    cap = cv2.VideoCapture(og_vid)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_limit = int(fps * 60)  

    # Define output video writer
    out = cv2.VideoWriter(
        "/Users/koshabbas/Desktop/work/stock_market/output_video.mp4", 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (width, height)
    )

    frame_count = 0

    # Process video frame by frame
    while frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV default) to RGB for model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO prediction on RGB frame
        results = model.predict(source=rgb_frame, conf=0.3, imgsz=512, verbose=False)

        # Draw boxes on RGB frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                label = f"{conf:.2f}"
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert RGB back to BGR for OpenCV video writer
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(bgr_frame)
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


execute_model()

# def main():
#     model = YOLO('/Users/koshabbas/Desktop/work/stock_market/detect/train_run165/weights/best.pt')
#     og_vid = "/Users/koshabbas/Desktop/work/stock_market/Trading 053025-Meeting Recording.mp4"
#     #model_on_vid(model,og_vid)
#     asses_performance(model)
# main()