from ultralytics import YOLO

# Load a fresh YOLOv8 model
model = YOLO('yolov8n.pt')

# Train on the combined dataset
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='combined_model'
)

print("Training complete! Model saved to runs/detect/combined_model/")