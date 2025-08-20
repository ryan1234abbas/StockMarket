import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


IMG_SIZE = 32
LABELS = ["HH", "HL", "LH", "LL"]
DATASET_DIR = "dataset"  
MODEL_PATH = "label_cnn.h5"


def load_multi_source_dataset(folders):
    """
    folders: list of base folders (e.g., ["templates", "templates_windows"])
    Returns: X, y for training
    """
    X, y = [], []

    for base_dir in folders:
        for idx, label in enumerate(LABELS):
            label_dir = os.path.join(base_dir, label)
            if not os.path.exists(label_dir):
                continue
            for file in os.listdir(label_dir):
                path = os.path.join(label_dir, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                X.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
                y.append(idx)

    X = np.array(X)
    y = to_categorical(y, num_classes=len(LABELS))
    return X, y

# Build model
def build_model():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,1)),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(LABELS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model

def train_model():
    folders = ["templates", "templates_windows"]  
    print("Loading dataset from multiple sources...")
    X, y = load_multi_source_dataset(folders)
    print(f"Dataset loaded: {len(X)} samples")

    model = build_model()
    print("Training model...")
    model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


# Test model
def test_model(image_path):
    if not os.path.exists(MODEL_PATH):
        print("No trained model found, training first...")
        model = train_model()
    else:
        model = load_model(MODEL_PATH)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found:", image_path)
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    preds = model.predict(img, verbose=0)[0]
    label_idx = np.argmax(preds)
    conf = preds[label_idx]
    print(f"Predicted label: {LABELS[label_idx]}, Confidence: {conf:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and test label CNN")
    parser.add_argument("--train", action="store_true", help="Train the CNN model")
    parser.add_argument("--test", type=str, help="Test a single image")

    args = parser.parse_args()

    if args.train:
        train_model()
    if args.test:
        test_model(args.test)

'''
Take a bunch of label screenshots on windows 
and retrain on macos

Then assess performance alongside TM and decide
'''